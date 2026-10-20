"""GPU-02: attempt every registered site, one resident model at a time.

A multi-hook forward is used only after its activation and SAE codes match
single-hook capture_features on one prompt. A mismatch falls back to
capture_features. Incomplete historical hashes are recomputed after load and
are not treated as a match.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from common import INCOMPLETE_SITES, free_gib, log, write_json


def _natural_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset_id") != "controlled6":
                count += 1
    return count


def _all_records(root: Path) -> list[dict[str, Any]]:
    from run_gpu import json_loads

    corpus = root / "private" / "corpus"
    rows = []
    for dataset_dir in sorted(
        path for path in corpus.iterdir() if (path / "records.jsonl").is_file()
    ):
        with (dataset_dir / "records.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json_loads(line))
    if len(rows) != 500:
        raise RuntimeError("Private corpus is not 500 groups")
    return rows


def _public_row(measured: dict, record: dict, site: dict, model_sha: str, sae_sha: str) -> dict:
    prompt = record["prompt"]
    public = {
        key: value
        for key, value in measured.items()
        if key
        in {
            "l0",
            "reconstruction_mse",
            "token_position",
            "capture_seconds",
            "elapsed_seconds",
            "peak_vram_bytes",
        }
    }
    public.update(
        {
            "site_id": site["site_id"],
            "sample_id": record["sample_id"],
            "prompt_group_id": record["prompt_group_id"],
            "dataset_id": record["dataset_id"],
            "usage_split": record["usage_split"],
            "loaded_model_sha256": model_sha,
            "loaded_sae_sha256": sae_sha,
            "hash_matches_historical": bool(site.get("model_sha256"))
            and model_sha == site.get("model_sha256")
            and sae_sha == site.get("sae_sha256"),
            "historical_hash_was_empty": site["site_id"] in INCOMPLETE_SITES
            or not site.get("model_sha256"),
            "text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(measured["token_ids"]).encode("utf-8")
            ).hexdigest(),
        }
    )
    return public


def _append(root: Path, site_id: str, public: dict, measured: dict) -> None:
    private = dict(public)
    private["token_ids"] = measured["token_ids"]
    private["nonzero_codes"] = measured["nonzero_codes"]
    path = root / "private" / f"{site_id.replace(':', '_')}_codes.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(private, ensure_ascii=False, allow_nan=False) + "\n")


def _measure_capture(adapter, sae, activation_site, tokens) -> dict:
    import time

    from saebase.api.features import capture_features

    started = time.perf_counter()
    snapshot = capture_features(adapter, sae, site=activation_site, tokens=tokens)
    nonzero = torch.nonzero(snapshot.codes, as_tuple=False).flatten()
    mse = float(
        torch.mean((snapshot.activation.float() - snapshot.reconstruction.float()) ** 2).item()
    )
    return {
        "l0": int(nonzero.numel()),
        "reconstruction_mse": mse,
        "token_position": int(tokens.shape[1] - 1),
        "token_ids": [int(token) for token in tokens[0].tolist()],
        "nonzero_codes": [
            [int(index), float(snapshot.codes[int(index)].item())] for index in nonzero.tolist()
        ],
        "capture_seconds": time.perf_counter() - started,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "snapshot": snapshot,
    }


def _codes_from_cached_activation(sae, activation, tokens) -> dict | None:
    """Encode a cached hook vector with the same helpers capture_features uses."""
    import time

    try:
        from saebase.api.features import (
            _capture_ln_stats,
            _check_vector,
            _decode,
            _sae_dimensions,
            _sae_input,
        )
    except Exception:
        return None
    started = time.perf_counter()
    d_in, d_sae = _sae_dimensions(sae)
    _check_vector("activation", activation, size=d_in)
    hidden = _sae_input(sae, activation)
    codes = sae.encode(hidden.unsqueeze(0))[0]
    _check_vector("codes", codes, size=d_sae)
    ln_mu, ln_std = _capture_ln_stats(sae)
    reconstruction = _decode(sae, codes, d_in, ln_mu=ln_mu, ln_std=ln_std)
    nonzero = torch.nonzero(codes, as_tuple=False).flatten()
    mse = float(torch.mean((activation.float() - reconstruction.float()) ** 2).item())
    return {
        "l0": int(nonzero.numel()),
        "reconstruction_mse": mse,
        "token_position": int(tokens.shape[1] - 1),
        "token_ids": [int(token) for token in tokens[0].tolist()],
        "nonzero_codes": [
            [int(index), float(codes[int(index)].item())] for index in nonzero.tolist()
        ],
        "capture_seconds": time.perf_counter() - started,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _multi_hook_parity(model, bundles, tokens) -> dict:
    from saebase.api import ActivationSite, ModelAdapter

    adapter = ModelAdapter(model)
    if adapter.backend != "transformer_lens":
        return {"status": "fallback_sequential", "reason": "backend has no shared hook cache"}
    names = [site["hook"] for site, _sae, _meta in bundles]
    reference = _measure_capture(
        adapter,
        bundles[0][1],
        ActivationSite(int(bundles[0][0]["layer"]), bundles[0][0]["hook"], position=-1),
        tokens,
    )
    with torch.no_grad():
        _logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    cached = cache[names[0]][0, -1]
    activation_ok = torch.allclose(cached, reference["snapshot"].activation, rtol=1e-4, atol=1e-4)
    encoded = (
        _codes_from_cached_activation(bundles[0][1], cached, tokens) if activation_ok else None
    )
    codes_ok = False
    if encoded is not None:
        left = {pair[0]: pair[1] for pair in encoded["nonzero_codes"]}
        right = {pair[0]: pair[1] for pair in reference["nonzero_codes"]}
        codes_ok = left.keys() == right.keys() and all(
            abs(left[key] - right[key]) <= 1e-4 for key in left
        )
    if activation_ok and codes_ok:
        return {"status": "passed", "backend": adapter.backend}
    return {
        "status": "fallback_sequential",
        "activation_match": bool(activation_ok),
        "code_match": bool(codes_ok),
    }


def _capture_sequential(root, model, bundle, records, done_ids) -> dict:
    import time

    from run_gpu import _exact_tokens
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.store.identity import sha256_state_dict

    site, sae, meta = bundle
    adapter = ModelAdapter(model)
    model_sha = sha256_state_dict(model)
    sae_sha = sha256_state_dict(sae)
    activation_site = ActivationSite(int(site["layer"]), site["hook"], position=-1)
    failures = []
    measured = 0
    started = time.perf_counter()
    for record in records:
        if record["sample_id"] in done_ids:
            continue
        tokens, error = _exact_tokens(model, record["prompt"])
        if error or tokens is None:
            failures.append({"sample_id": record["sample_id"], "error": error})
            continue
        try:
            row = _measure_capture(adapter, sae, activation_site, tokens)
        except Exception as exc:
            failures.append(
                {
                    "sample_id": record["sample_id"],
                    "error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:240],
                }
            )
            continue
        public = _public_row(row, record, site, model_sha, sae_sha)
        _append(root, site["site_id"], public, row)
        measured += 1
    return _summary(
        site,
        meta,
        model_sha,
        sae_sha,
        measured,
        len(done_ids),
        failures,
        time.perf_counter() - started,
    )


def _capture_cached(root, model, bundles, records, done_by_site) -> list[dict]:
    import time

    from run_gpu import _exact_tokens
    from saebase.store.identity import sha256_state_dict

    model_sha = sha256_state_dict(model)
    identities = [(site, sae, meta, sha256_state_dict(sae)) for site, sae, meta in bundles]
    names = [site["hook"] for site, _sae, _meta, _sha in identities]
    failures = {site["site_id"]: [] for site, *_rest in identities}
    measured = {site["site_id"]: 0 for site, *_rest in identities}
    started = time.perf_counter()
    for record in records:
        needed = [
            row for row in identities if record["sample_id"] not in done_by_site[row[0]["site_id"]]
        ]
        if not needed:
            continue
        tokens, error = _exact_tokens(model, record["prompt"])
        if error or tokens is None:
            for site, *_rest in needed:
                failures[site["site_id"]].append({"sample_id": record["sample_id"], "error": error})
            continue
        with torch.no_grad():
            _logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
        for site, sae, _meta, sae_sha in needed:
            cached = cache[site["hook"]][0, -1]
            row = _codes_from_cached_activation(sae, cached, tokens)
            if row is None:
                failures[site["site_id"]].append(
                    {"sample_id": record["sample_id"], "error": "cached_encode_unavailable"}
                )
                continue
            public = _public_row(row, record, site, model_sha, sae_sha)
            _append(root, site["site_id"], public, row)
            measured[site["site_id"]] += 1
    elapsed = time.perf_counter() - started
    summaries = []
    for site, _sae, meta, sae_sha in identities:
        summaries.append(
            _summary(
                site,
                meta,
                model_sha,
                sae_sha,
                measured[site["site_id"]],
                len(done_by_site[site["site_id"]]),
                failures[site["site_id"]],
                elapsed,
            )
        )
    return summaries


def _summary(site, meta, model_sha, sae_sha, new_rows, skipped, failures, elapsed) -> dict:
    total = new_rows + skipped
    return {
        "site_id": site["site_id"],
        "status": "passed" if total == 500 and not failures else ("partial" if total else "failed"),
        "hook": site["hook"],
        "layer": site["layer"],
        "loaded_model_name": meta.get("loaded_model_name"),
        "loaded_model_sha256": model_sha,
        "loaded_sae_sha256": sae_sha,
        "historical_model_sha256": site.get("model_sha256"),
        "historical_sae_sha256": site.get("sae_sha256"),
        "hash_matches_historical": bool(site.get("model_sha256"))
        and model_sha == site.get("model_sha256")
        and sae_sha == site.get("sae_sha256"),
        "recomputed_identity": site["site_id"] in INCOMPLETE_SITES or not site.get("model_sha256"),
        "measured": total,
        "failed_count": len(failures),
        "failed_sample_ids": [row["sample_id"] for row in failures[:20]],
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
    }


def _identity_from_private(root: Path, site_id: str) -> dict:
    path = root / "private" / f"{site_id.replace(':', '_')}_codes.jsonl"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        line = stream.readline()
    if not line.strip():
        return {}
    row = json.loads(line)
    empty = bool(row.get("historical_hash_was_empty"))
    return {
        "loaded_model_sha256": row.get("loaded_model_sha256"),
        "loaded_sae_sha256": row.get("loaded_sae_sha256"),
        "hash_matches_historical": row.get("hash_matches_historical"),
        "historical_hash_was_empty": row.get("historical_hash_was_empty"),
        "recomputed_identity": empty or bool(row.get("recomputed_identity")),
    }


def _blocked(site_id: str, reason: str) -> dict:
    return {"site_id": site_id, "status": "blocked", "measured": 0, "reason": reason[:400]}


def run_all_sites(root: Path, pilot: dict) -> dict:
    from collections import defaultdict

    from run_gpu import _exact_tokens, _matrix_access
    from saebase.sae.loading import load_model_for_profile, load_sae_for_profile

    if not pilot.get("go"):
        decision = {
            "status": "partial",
            "reason": "GPU-01 did not pass on both the small site and the deep site, so the 500-group capture stays stopped.",
        }
        write_json(root / "GPU-02" / "decision.json", decision)
        return decision
    records = _all_records(root)
    from backend.core import load_sites

    sites = load_sites()
    access = _matrix_access()
    grouped: dict[str, list] = defaultdict(list)
    for site in sites:
        grouped[site["model"]].append(site)
    ordered = sorted(
        grouped,
        key=lambda model: (
            0 if any(access.get(site["site_id"]) == "open" for site in grouped[model]) else 1,
            model,
        ),
    )
    results = []
    parity_notes = []
    stopped_for_disk = False
    prior = {}
    prior_path = root / "GPU-02" / "decision.json"
    if prior_path.is_file():
        for row in json.loads(prior_path.read_text(encoding="utf-8")).get("site_results", []):
            if row.get("site_id"):
                prior[row["site_id"]] = row
    for model_name in ordered:
        group = grouped[model_name]
        pending = []
        for site in group:
            private = root / "private" / f"{site['site_id'].replace(':', '_')}_codes.jsonl"
            if _natural_count(private) >= 500:
                kept = dict(prior.get(site["site_id"], {}))
                kept.update(
                    {
                        "site_id": site["site_id"],
                        "status": "passed",
                        "measured": 500,
                        "reused": True,
                        "access": access.get(site["site_id"]),
                    }
                )
                if not kept.get("loaded_model_sha256"):
                    kept.update(_identity_from_private(root, site["site_id"]))
                results.append(kept)
                log(root, f"GPU-02 reuse {site['site_id']}")
            else:
                pending.append(site)
        if not pending:
            continue
        if free_gib(root) < 40:
            stopped_for_disk = True
            for site in pending:
                results.append(
                    _blocked(site["site_id"], "disk free space below 40 GiB; new downloads stopped")
                )
            continue
        log(root, f"GPU-02 loading model {model_name} for {len(pending)} sites")
        try:
            model, loaded_name = load_model_for_profile(pending[0]["sae"], device="cuda")
        except Exception as exc:
            reason = type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]
            for site in pending:
                results.append(_blocked(site["site_id"], reason))
            log(root, f"GPU-02 {model_name} blocked {reason}")
            continue
        bundles = []
        try:
            for site in pending:
                try:
                    sae, meta = load_sae_for_profile(site["sae"], int(site["layer"]), device="cuda")
                except Exception as exc:
                    results.append(
                        _blocked(
                            site["site_id"],
                            type(exc).__name__ + ": " + str(exc).splitlines()[0][:300],
                        )
                    )
                    continue
                meta = {**meta, "loaded_model_name": loaded_name}
                bundles.append((site, sae, meta))
            if not bundles:
                continue
            probe_tokens = None
            for record in records:
                probe_tokens, error = _exact_tokens(model, record["prompt"])
                if probe_tokens is not None:
                    break
            parity = {"status": "fallback_sequential", "reason": "no in-context probe prompt"}
            if probe_tokens is not None:
                try:
                    parity = _multi_hook_parity(model, bundles, probe_tokens)
                except Exception as exc:
                    parity = {
                        "status": "fallback_sequential",
                        "reason": type(exc).__name__ + ": " + str(exc).splitlines()[0][:200],
                    }
            parity_notes.append({"model": model_name, **parity})
            log(root, f"GPU-02 {model_name} hook parity {parity['status']}")
            if parity["status"] == "passed":
                done_by_site = {}
                for site, _sae, _meta in bundles:
                    done_by_site[site["site_id"]] = _done_ids(root, site["site_id"])
                results.extend(_capture_cached(root, model, bundles, records, done_by_site))
            else:
                for bundle in bundles:
                    done = _done_ids(root, bundle[0]["site_id"])
                    results.append(_capture_sequential(root, model, bundle, records, done))
                    log(
                        root,
                        f"GPU-02 {bundle[0]['site_id']} measured={results[-1].get('measured')} failed={results[-1].get('failed_count')}",
                    )
        finally:
            del model
            for bundle in bundles:
                del bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    attempted = [row for row in results if not row.get("reused")]
    passed = [row for row in results if row.get("status") == "passed"]
    decision = {
        "status": "passed" if len(passed) == 48 else "partial",
        "denominator_sites": 48,
        "sites_passed": len(passed),
        "sites_attempted_this_run": len(attempted),
        "disk_stop": stopped_for_disk,
        "hook_parity": parity_notes,
        "site_results": results,
        "reachable_open_sites": [row["site_id"] for row in passed],
        "reason": (
            f"{len(passed)}/48 sites have 500 captured groups. "
            "Failures and gated download errors stay in the denominator. "
            "These captures are not Stage D 64-group results. "
            "Recomputed hashes on the three historically empty sites are not historical matches. "
            "A loaded hash that differs from a non-empty historical hash is not inserted as that historical identity."
        ),
    }
    write_json(root / "GPU-02" / "decision.json", decision)
    _write_coverage(root, results, access)
    log(root, f"GPU-02 finished passed={len(passed)}/48")
    return decision


def _done_ids(root: Path, site_id: str) -> set[str]:
    path = root / "private" / f"{site_id.replace(':', '_')}_codes.jsonl"
    found = set()
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("dataset_id") != "controlled6":
                    found.add(row["sample_id"])
    return found


def _write_coverage(root: Path, results: list[dict], access: dict[str, str]) -> None:
    path = root / "GPU-02" / "site_coverage.csv"
    fields = [
        "site_id",
        "access",
        "status",
        "measured",
        "failed_count",
        "hash_matches_historical",
        "recomputed_identity",
        "elapsed_seconds",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {key: row.get(key, "") for key in fields}
                | {"access": access.get(row["site_id"], row.get("access", ""))}
            )
    write_json(
        root / "GPU-02" / "source_digests.json",
        [
            {
                "site_id": row["site_id"],
                "loaded_model_sha256": row.get("loaded_model_sha256"),
                "loaded_sae_sha256": row.get("loaded_sae_sha256"),
                "hash_matches_historical": row.get("hash_matches_historical"),
            }
            for row in results
        ],
    )
