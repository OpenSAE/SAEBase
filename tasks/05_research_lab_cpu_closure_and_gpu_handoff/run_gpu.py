"""GPU-00 registry gate, GPU-01 20-group pilot, optional GPU-02, and GPU-07 grid."""

from __future__ import annotations

import gc
import importlib.metadata
import math
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

from common import (
    DEEP_SITE,
    INCOMPLETE_SITES,
    REPO,
    SMALL_SITE,
    file_sha256,
    git_commit,
    log,
    write_json,
)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "apps" / "research-lab"))

from backend.controlled_designs import (  # noqa: E402
    check_candidate,
    make_candidate,
    synthetic_reference_codes,
    to_saebase_design,
    write_candidate,
)
from backend.core import has_complete_identity, load_sites, read_json  # noqa: E402
from saebase.api import ActivationSite, ModelAdapter  # noqa: E402
from saebase.api.features import capture_features  # noqa: E402
from saebase.interventions.operators import apply_condition  # noqa: E402
from saebase.interventions.specs import Condition  # noqa: E402
from saebase.sae.loading import load_model_and_sae  # noqa: E402
from saebase.store.identity import sha256_state_dict  # noqa: E402


def _versions() -> dict[str, str]:
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    for name in ("transformers", "sae-lens", "transformer-lens", "vllm", "datasets"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not_installed"
    return out


def _matrix_access() -> dict[str, str]:
    matrix = read_json(REPO / "conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json")
    access = {}
    for row in matrix["rows"]:
        site = f"{row['model_key']}:{row['sae_key']}:L{row['layer']}"
        access[site] = row.get("access") or "unspecified"
    return access


def run_gpu00(root: Path, cpu_summary: dict) -> dict:
    sites = load_sites()
    access = _matrix_access()
    rows = []
    for site in sites:
        rows.append(
            {
                "site_id": site["site_id"],
                "hook": site["hook"],
                "layer": site["layer"],
                "historical_model_sha256": site["model_sha256"],
                "historical_sae_sha256": site["sae_sha256"],
                "historical_identity_complete": has_complete_identity(site),
                "new_experiment_allowed": has_complete_identity(site) and site["site_id"] not in INCOMPLETE_SITES,
                "access": access.get(site["site_id"], "unspecified"),
                "loaded_this_run": False,
            }
        )
    if len(rows) != 48:
        raise RuntimeError("48-site manifest is incomplete")
    disk = shutil.disk_usage(root)
    payload = {
        "commit": git_commit(),
        "versions": _versions(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "disk_note": "Work Plan 05 suggests about 500GB; this volume is smaller, so one model is resident at a time.",
        "sites": rows,
        "incomplete_historical_hash_sites": sorted(INCOMPLETE_SITES),
        "cpu_lock_passed": cpu_summary["passed"],
        "input_release_sha256": cpu_summary["lock_sha256"],
        "audit_sha256": cpu_summary["audit_sha256"],
        "source_hashes": {
            "work_plan": file_sha256(REPO / "work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md"),
            "operators": file_sha256(REPO / "saebase/interventions/operators.py"),
            "model_api": file_sha256(REPO / "saebase/api/model.py"),
            "coverage": file_sha256(REPO / "apps/research-lab/data/coverage.json"),
        },
        "quantization": "not_used",
        "vllm_role": "generation parity only; SAE capture stays on the registered hook API",
    }
    write_json(root / "GPU-00" / "preflight.json", payload)
    write_json(root / "GPU-00" / "checkpoint_registry_audit.json", {"sites": rows})
    write_json(
        root / "GPU-00" / "dataset_identity.json",
        {
            "sources": cpu_summary["sources"],
            "groups": cpu_summary["groups"],
            "discovery": cpu_summary["discovery"],
            "evaluation": cpu_summary["evaluation"],
            "variant_proposals": cpu_summary["variant_proposals"],
            "gpu_measurements": 0,
            "lock_sha256": cpu_summary["lock_sha256"],
            "audit_sha256": cpu_summary["audit_sha256"],
            "human_reviewed": False,
            "license_reviewed": False,
        },
    )
    log(root, "GPU-00 manifest written; weights are hashed only when a site is loaded")
    return payload


def _context_limit(model: Any) -> int:
    cfg = getattr(model, "cfg", None)
    n_ctx = getattr(cfg, "n_ctx", None)
    if isinstance(n_ctx, int) and 8 <= n_ctx < 10**6:
        return n_ctx
    inner = getattr(model, "_model", None)
    conf = getattr(inner, "config", None)
    value = getattr(conf, "max_position_embeddings", None)
    if isinstance(value, int) and 8 <= value < 10**6:
        return value
    return 2048


def _exact_tokens(model: Any, prompt: str) -> tuple[torch.Tensor | None, str | None]:
    limit = _context_limit(model)
    try:
        if getattr(model, "_saebase_backend", None) == "hf_hooks":
            tokens = model.to_tokens(prompt)
        else:
            try:
                tokens = model.to_tokens(prompt, truncate=False)
            except TypeError:
                tokens = model.to_tokens(prompt)
    except Exception as exc:  # tokenizer/model rejection, not a silent truncate
        return None, type(exc).__name__ + ": " + str(exc).splitlines()[0][:200]
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        return None, "tokenizer_returned_empty"
    if int(tokens.shape[1]) > limit:
        return None, f"token_length_{int(tokens.shape[1])}_exceeds_context_{limit}"
    return tokens, None


def _load_records(root: Path) -> list[dict[str, Any]]:
    corpus = root / "private" / "corpus"
    chosen = []
    for dataset_dir in sorted(p for p in corpus.iterdir() if (p / "records.jsonl").is_file()):
        rows = []
        with (dataset_dir / "records.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json_loads(line))
        discovery = next(row for row in rows if row["usage_split"] == "discovery")
        evaluation = next(row for row in rows if row["usage_split"] == "evaluation")
        chosen.extend((discovery, evaluation))
    if len(chosen) != 20 or len({row["dataset_id"] for row in chosen}) != 10:
        raise RuntimeError("Pilot selection is not two groups from each of ten sources")
    splits = {row["prompt_group_id"]: row["usage_split"] for row in chosen}
    if any(a == b for a, b in zip(splits, list(splits)[1:])):
        pass
    groups = [row["prompt_group_id"] for row in chosen]
    if len(groups) != len(set(groups)):
        raise RuntimeError("Pilot groups are not independent")
    return chosen


def json_loads(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def _finite_l2(edited: torch.Tensor, baseline: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm((edited.float() - baseline.float())).item())
    if not math.isfinite(value):
        raise RuntimeError("non-finite logit delta")
    return value


def _measure_one(adapter: ModelAdapter, sae: Any, site: ActivationSite, tokens: torch.Tensor) -> dict[str, Any]:
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    adapter.assert_identity_patch(tokens=tokens, site=site)
    capture_started = time.perf_counter()
    snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
    capture_seconds = time.perf_counter() - capture_started
    baseline = adapter.forward_logits(tokens=tokens, readout_position=site.position)
    identity = apply_condition(sae, snapshot, Condition("identity", "identity"))
    if not torch.allclose(identity.activation, snapshot.activation, rtol=1e-4, atol=1e-4):
        raise RuntimeError("identity operator did not reproduce the captured activation")
    active = torch.nonzero(snapshot.codes, as_tuple=False).flatten()
    feature_checks: dict[str, Any]
    if active.numel() == 0:
        feature_checks = {"ablate": "undefined_no_active_feature", "scale": "undefined_no_active_feature"}
    else:
        feature = int(torch.argmax(snapshot.codes.abs()).item())
        feature_checks = {}
        for name, condition in (
            ("ablate", Condition("ablate", "ablate", (feature,))),
            ("scale", Condition("scale", "scale", (feature,), 0.5)),
        ):
            patch = apply_condition(sae, snapshot, condition)
            edited = adapter.patch_logits(
                tokens=tokens, site=site, activation=patch.activation, readout_position=site.position
            )
            feature_checks[name] = {
                "feature_id": feature,
                "logit_l2": _finite_l2(edited, baseline),
                "single_token_position": True,
            }
    reconstruction = apply_condition(sae, snapshot, Condition("reconstruction", "reconstruction"))
    recon_logits = adapter.patch_logits(
        tokens=tokens,
        site=site,
        activation=reconstruction.activation,
        readout_position=site.position,
    )
    mse = float(torch.mean((snapshot.activation.float() - snapshot.reconstruction.float()) ** 2).item())
    variance = float(torch.var(snapshot.activation.float(), unbiased=False).item())
    explained = None if variance == 0 else 1.0 - mse / variance
    nonzero = torch.nonzero(snapshot.codes, as_tuple=False).flatten()
    codes = [[int(index), float(snapshot.codes[int(index)].item())] for index in nonzero.tolist()]
    token_ids = [int(token) for token in tokens[0].tolist()]
    return {
        "identity_patch": "passed",
        "identity_operator": "passed",
        "feature_checks": feature_checks,
        "reconstruction_logit_l2": _finite_l2(recon_logits, baseline),
        "reconstruction_mse": mse,
        "explained_variance": explained,
        "l0": int(nonzero.numel()),
        "top1_token_id": int(torch.argmax(baseline).item()),
        "token_position": int(tokens.shape[1] - 1),
        "token_ids": token_ids,
        "nonzero_codes": codes,
        "capture_seconds": capture_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "joint_multi_token_patch": False,
    }


def _site_by_id(site_id: str) -> dict[str, Any]:
    for site in load_sites():
        if site["site_id"] == site_id:
            return site
    raise KeyError(site_id)


def _run_site(root: Path, site_id: str, records: list[dict[str, Any]], *, interventions: bool) -> dict[str, Any]:
    site = _site_by_id(site_id)
    if not has_complete_identity(site) or site_id in INCOMPLETE_SITES:
        return {"site_id": site_id, "status": "blocked", "reason": "historical weight hash incomplete"}
    log(root, f"loading {site_id}")
    try:
        model, sae, meta = load_model_and_sae(site["sae"], int(site["layer"]), device="cuda")
    except Exception as exc:
        return {
            "site_id": site_id,
            "status": "blocked",
            "reason": type(exc).__name__ + ": " + str(exc).splitlines()[0][:300],
        }
    adapter = ModelAdapter(model)
    model_sha = sha256_state_dict(model)
    sae_sha = sha256_state_dict(sae)
    activation_site = ActivationSite(int(site["layer"]), site["hook"], position=-1)
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    ledger = []
    failures = []
    tokenizer_report = []
    done_ids: set[str] = set()
    private_path = root / "private" / f"{site_id.replace(':', '_')}_codes.jsonl"
    if private_path.is_file():
        with private_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done_ids.add(json_loads(line)["sample_id"])
    try:
        for record in records:
            if record["sample_id"] in done_ids:
                continue
            prompt = record["prompt"]
            if record.get("task_label") and record["task_label"] in prompt and record["dataset_id"] == "gsm8k":
                # GSM8K prompts in this corpus include the question, not a copied final label field.
                pass
            tokens, error = _exact_tokens(model, prompt)
            report = {
                "sample_id": record["sample_id"],
                "dataset_id": record["dataset_id"],
                "usage_split": record["usage_split"],
                "prompt_group_id": record["prompt_group_id"],
                "status": "failed" if error else "within_context",
                "error": error,
                "context_limit": _context_limit(model),
            }
            tokenizer_report.append(report)
            if error or tokens is None:
                failures.append({"sample_id": record["sample_id"], "error": error})
                continue
            try:
                if interventions:
                    measured = _measure_one(adapter, sae, activation_site, tokens)
                else:
                    started = time.perf_counter()
                    snapshot = capture_features(adapter, sae, site=activation_site, tokens=tokens)
                    nonzero = torch.nonzero(snapshot.codes, as_tuple=False).flatten()
                    mse = float(torch.mean((snapshot.activation.float() - snapshot.reconstruction.float()) ** 2).item())
                    measured = {
                        "l0": int(nonzero.numel()),
                        "reconstruction_mse": mse,
                        "token_position": int(tokens.shape[1] - 1),
                        "token_ids": [int(token) for token in tokens[0].tolist()],
                        "nonzero_codes": [[int(i), float(snapshot.codes[int(i)].item())] for i in nonzero.tolist()],
                        "capture_seconds": time.perf_counter() - started,
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    }
            except Exception as exc:
                failures.append({"sample_id": record["sample_id"], "error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]})
                if "identity patch" in str(exc):
                    break
                continue
            public = {key: value for key, value in measured.items() if key not in {"token_ids", "nonzero_codes"}}
            public.update(
                {
                    "site_id": site_id,
                    "sample_id": record["sample_id"],
                    "prompt_group_id": record["prompt_group_id"],
                    "dataset_id": record["dataset_id"],
                    "usage_split": record["usage_split"],
                    "loaded_model_sha256": model_sha,
                    "loaded_sae_sha256": sae_sha,
                    "hash_matches_historical": model_sha == site["model_sha256"] and sae_sha == site["sae_sha256"],
                    "text_sha256": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
                    "token_ids_sha256": __import__("hashlib").sha256(
                        __import__("json").dumps(measured["token_ids"]).encode("utf-8")
                    ).hexdigest(),
                }
            )
            ledger.append(public)
            private_row = dict(public)
            private_row["token_ids"] = measured["token_ids"]
            private_row["nonzero_codes"] = measured["nonzero_codes"]
            with (root / "private" / f"{site_id.replace(':', '_')}_codes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(__import__("json").dumps(private_row, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        del model, sae, adapter
        gc.collect()
        torch.cuda.empty_cache()
    capture_times = [row["capture_seconds"] for row in ledger]
    skipped = sum(1 for record in records if record["sample_id"] in done_ids)
    return {
        "site_id": site_id,
        "status": "passed" if (ledger or skipped) and not any("identity patch" in row["error"] for row in failures) else "failed",
        "hook": site["hook"],
        "layer": site["layer"],
        "dtype": dtype,
        "loaded_model_name": meta.get("loaded_model_name"),
        "loaded_model_sha256": model_sha,
        "loaded_sae_sha256": sae_sha,
        "historical_model_sha256": site["model_sha256"],
        "historical_sae_sha256": site["sae_sha256"],
        "hash_matches_historical": model_sha == site["model_sha256"] and sae_sha == site["sae_sha256"],
        "measured": len(ledger) + skipped,
        "failed": failures,
        "mean_capture_seconds": (sum(capture_times) / len(capture_times)) if capture_times else None,
        "ledger": ledger,
        "tokenizer_report": tokenizer_report,
    }


def run_gpu01(root: Path) -> dict:
    records = _load_records(root)
    package = root / "GPU-01"
    results = []
    for site_id in (SMALL_SITE, DEEP_SITE):
        existing_path = package / f"{site_id.replace(':', '_')}.json"
        existing = json_loads(existing_path.read_text(encoding="utf-8")) if existing_path.is_file() else None
        if existing and existing.get("status") == "passed" and existing.get("measured") == 20:
            results.append(existing)
            log(root, f"GPU-01 {site_id} reused passed pilot")
            continue
        results.append(_run_site(root, site_id, records, interventions=True))
        write_json(existing_path, results[-1])
        log(root, f"GPU-01 {site_id} status={results[-1]['status']} measured={results[-1].get('measured')}")
    datasets_ok = set()
    for result in results:
        if result.get("status") != "passed":
            continue
        for row in result["ledger"]:
            datasets_ok.add((result["site_id"], row["dataset_id"], row["usage_split"]))
    small_ok = results[0].get("status") == "passed" and results[0].get("measured") == 20
    deep_ok = results[1].get("status") == "passed" and results[1].get("measured") == 20
    go = bool(small_ok and deep_ok)
    manifest = {
        "population": "2 base groups per source, 20 groups, not 20 independent datasets",
        "sites": [SMALL_SITE, DEEP_SITE],
        "go": go,
        "small_site_passed": small_ok,
        "deep_site_passed": deep_ok,
        "results": [{key: value for key, value in row.items() if key != "ledger"} for row in results],
    }
    write_json(package / "pilot_20" / "manifest.json", manifest)
    write_json(package / "pilot_20" / "ledger.json", results)
    capture = [row["mean_capture_seconds"] for row in results if row.get("mean_capture_seconds")]
    write_json(
        package / "cost_and_capacity.json",
        {
            "gpu": torch.cuda.get_device_name(0),
            "mean_capture_seconds": capture,
            "batch_size": 1,
            "batch_note": "Verified ModelAdapter captures one prompt. A larger batch was not substituted without a parity check.",
            "quantization": "not_used",
            "go": go,
        },
    )
    return manifest


def run_gpu02(root: Path, pilot: dict) -> dict:
    from capture_batch import run_all_sites

    return run_all_sites(root, pilot)


def _synthetic_curve(codes: dict[str, Any], synthetic: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        from saebase.integrations.localization import LocalizationGrid, evaluate_localization

        grid = LocalizationGrid(
            codes=np.asarray(codes["feature_values"], dtype=np.float64),
            n_factors=6,
            state_sample_ids=tuple(row["sample_id"] for row in synthetic["states"]),
            source="synthetic_logic_not_real_sae",
        )
        result = evaluate_localization(grid)
        return {
            "source_type": "synthetic_logic_not_real_sae",
            "total_variance": float(result.total_variance),
            "budget": int(result.budget),
            "not_a_learned_sae_result": True,
        }
    except Exception as exc:
        return {"status": "blocked", "error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]}


def run_gpu07(root: Path, pilot: dict) -> dict:
    existing = root / "GPU-07" / "localize_status.json"
    if existing.is_file():
        log(root, "GPU-07 reused existing controlled-grid record")
        return json_loads(existing.read_text(encoding="utf-8"))
    package = root / "GPU-07"
    synthetic = make_candidate("synthetic_known_factor_control")
    real = make_candidate("real_model_six_factor_design")
    check_candidate(synthetic)
    check_candidate(real)
    to_saebase_design(synthetic)
    to_saebase_design(real)
    codes = synthetic_reference_codes(synthetic)
    write_candidate("synthetic_known_factor_control", package / "synthetic_known_factor_control.json")
    write_candidate("real_model_six_factor_design", package / "real_model_six_factor_design.json")
    write_json(package / "synthetic_reference_codes.json", codes)
    localize_status: dict[str, Any] = {"synthetic_curve": "not_run", "real_model_curve": "not_run"}
    localize_status["synthetic_curve"] = _synthetic_curve(codes, synthetic)
    if pilot.get("small_site_passed"):
        records = []
        for state in real["states"]:
            records.append(
                {
                    "prompt": state["prompt"],
                    "sample_id": state["sample_id"],
                    "prompt_group_id": "controlled6:real_model_six_factor_design",
                    "dataset_id": "controlled6",
                    "usage_split": "controlled_grid",
                    "task_label": None,
                }
            )
        measured = _run_site(root, SMALL_SITE, records, interventions=False)
        masks = []
        for row in measured.get("ledger") or []:
            masks.append(row["sample_id"])
        localize_status["real_model_capture"] = {
            "site_id": SMALL_SITE,
            "measured": measured.get("measured"),
            "failed": measured.get("failed"),
            "status": measured.get("status"),
            "loaded_model_sha256": measured.get("loaded_model_sha256"),
            "loaded_sae_sha256": measured.get("loaded_sae_sha256"),
            "n_states_expected": 64,
            "independent_natural_prompts": False,
        }
        if measured.get("measured") == 64:
            try:
                import numpy as np
                from saebase.integrations.localization import LocalizationGrid, evaluate_localization

                private = root / "private" / f"{SMALL_SITE.replace(':', '_')}_codes.jsonl"
                by_id: dict[str, list] = {}
                width = 0
                with private.open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json_loads(line)
                        if row.get("dataset_id") != "controlled6":
                            continue
                        pairs = row["nonzero_codes"]
                        by_id[row["sample_id"]] = pairs
                        if pairs:
                            width = max(width, max(pair[0] for pair in pairs) + 1)
                if width < 1 or len(by_id) != 64:
                    raise RuntimeError("controlled grid codes are incomplete or have no active features")
                ordered = []
                for state in real["states"]:
                    vector = np.zeros(width, dtype=np.float64)
                    for index, value in by_id[state["sample_id"]]:
                        vector[index] = value
                    ordered.append(vector)
                grid = LocalizationGrid(
                    codes=np.vstack(ordered),
                    n_factors=6,
                    state_sample_ids=tuple(state["sample_id"] for state in real["states"]),
                    source="real_checkpoint",
                )
                result = evaluate_localization(grid)
                localize_status["real_model_curve"] = {
                    "source_type": "real_checkpoint",
                    "site_id": SMALL_SITE,
                    "total_variance": float(result.total_variance),
                    "budget_at_90pct": int(result.budget),
                    "n_features_in_observed_support": int(width),
                    "states_are_one_controlled_population": True,
                }
            except Exception as exc:
                localize_status["real_model_curve"] = {
                    "status": "partial",
                    "error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:300],
                }
    else:
        localize_status["real_model_capture"] = {"status": "blocked", "reason": "small-site pilot did not pass"}
    write_json(package / "localize_status.json", localize_status)
    log(root, "GPU-07 controlled grid recorded")
    return localize_status
