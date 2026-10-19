"""GPU-04 interventions, GPU-05 SAR, GPU-06 RSS, and the Localize support check.

Estimators stay in the original sar.py, rss.py and sae_localize. A coalition
batch is used only after its scalar readouts match sequential patch_logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import REPO, log, write_json

DEEP_SITES = (
    "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14",
    "gemma1b:gemma1b_scope_resid:L13",
    "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22",
    "gemma3_4b_it:gemma3_4b_it_scope_resid:L22",
    "llama31_8b:llama31_8b_lxr_8x:L21",
    "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16",
)
SAR_SITES = (
    "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14",
    "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22",
    "llama31_8b:llama31_8b_lxr_8x:L21",
)
K = 16
M = 384
N_FIT = 256
SEED = 20261018


def _coverage_ok() -> None:
    coverage = json.loads((REPO / "apps/research-lab/data/coverage.json").read_text(encoding="utf-8"))
    if tuple(coverage["deep"]) != DEEP_SITES or tuple(coverage["sar"]) != SAR_SITES:
        raise RuntimeError("coverage.json deep or SAR sites differ from the pinned lists")


def _gpu02_passed(root: Path) -> set[str]:
    decision = json.loads((root / "GPU-02" / "decision.json").read_text(encoding="utf-8"))
    return {row["site_id"] for row in decision.get("site_results", []) if row.get("status") == "passed" and row.get("measured") == 500}


def _records(root: Path) -> list[dict[str, Any]]:
    from capture_batch import _all_records

    return _all_records(root)


def _site(site_id: str) -> dict[str, Any]:
    from run_gpu import _site_by_id

    return _site_by_id(site_id)


def _load_pair(site: dict):
    from saebase.sae.loading import load_model_and_sae

    return load_model_and_sae(site["sae"], int(site["layer"]), device="cuda")


def _paper(path: str):
    from saebase.integrations.sources import load_paper_module

    kind = "sar" if path.endswith("sar.py") else "rss"
    return load_paper_module(path, paper=kind)


def run_gpu04(root: Path) -> dict:
    """At least one clean/edited pair per loaded deep site. POST /api/jobs stays closed."""
    import gc

    from run_gpu import _exact_tokens, _measure_one
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.api.features import capture_features
    from saebase.interventions.operators import apply_condition
    from saebase.interventions.specs import Condition
    from saebase.store.identity import sha256_state_dict

    _coverage_ok()
    passed = _gpu02_passed(root)
    records = [row for row in _records(root) if row["usage_split"] == "evaluation"]
    prompt_row = records[0]
    already = {}
    prior_path = root / "GPU-04" / "decision.json"
    if prior_path.is_file():
        for row in json.loads(prior_path.read_text(encoding="utf-8")).get("pairs", []):
            if row.get("status") == "passed" and row.get("site_id"):
                already[row["site_id"]] = row
    pairs = []
    for site_id in DEEP_SITES:
        if site_id in already:
            pairs.append(already[site_id])
            continue
        if site_id not in passed:
            pairs.append({"site_id": site_id, "status": "not_started", "reason": "site has no 500-group capture"})
            continue
        site = _site(site_id)
        if "mlp" not in site["hook"] and site_id.startswith("mistral"):
            pairs.append({"site_id": site_id, "status": "failed", "reason": "Mistral site lost its MLP hook"})
            continue
        log(root, f"GPU-04 {site_id}")
        try:
            model, sae, meta = _load_pair(site)
        except Exception as exc:
            pairs.append({"site_id": site_id, "status": "blocked", "reason": type(exc).__name__ + ": " + str(exc).splitlines()[0][:240]})
            continue
        try:
            adapter = ModelAdapter(model)
            tokens, error = _exact_tokens(model, prompt_row["prompt"])
            if error or tokens is None:
                pairs.append({"site_id": site_id, "status": "failed", "reason": error})
                continue
            measured = _measure_one(
                adapter, sae, ActivationSite(int(site["layer"]), site["hook"], position=-1), tokens
            )
            snapshot = capture_features(
                adapter, sae, site=ActivationSite(int(site["layer"]), site["hook"], position=-1), tokens=tokens
            )
            active = [int(index) for index in torch.argsort(snapshot.codes.abs(), descending=True)[:2].tolist()]
            extra: dict[str, Any] = {}
            if len(active) < 1:
                extra = {"clamp": "undefined_no_active_feature", "add": "undefined_no_active_feature", "coalition": "undefined_no_active_feature"}
            else:
                feature = active[0]
                conditions = [
                    ("clamp", Condition("clamp", "clamp", (feature,), 1.0)),
                    ("add", Condition("add", "add", (feature,), 1.0)),
                ]
                if len(active) >= 2:
                    conditions.append(("coalition", Condition("coalition", "coalition", (active[0], active[1]), retained_ids=(active[0],))))
                else:
                    extra["coalition"] = "undefined_fewer_than_two_active_features"
                baseline = adapter.forward_logits(tokens=tokens)
                for name, condition in conditions:
                    patch = apply_condition(sae, snapshot, condition)
                    edited = adapter.patch_logits(tokens=tokens, site=snapshot.site, activation=patch.activation)
                    extra[name] = {"feature_ids": list(condition.feature_ids), "logit_l2": float(torch.linalg.vector_norm(edited.float() - baseline.float()).item())}
            pairs.append(
                {
                    "site_id": site_id,
                    "status": "passed",
                    "hook": site["hook"],
                    "prompt_group_id": prompt_row["prompt_group_id"],
                    "text_stored": False,
                    "loaded_model_sha256": sha256_state_dict(model),
                    "loaded_sae_sha256": sha256_state_dict(sae),
                    "identity_patch": measured["identity_patch"],
                    "feature_checks": measured["feature_checks"],
                    "extra_operators": extra,
                    "joint_multi_token_patch": False,
                    "generation": "full_prefix_reference",
                    "loaded_model_name": meta.get("loaded_model_name"),
                }
            )
        except Exception as exc:
            pairs.append({"site_id": site_id, "status": "failed", "reason": type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]})
        finally:
            del model, sae
            gc.collect()
            torch.cuda.empty_cache()
    done = [row for row in pairs if row.get("status") == "passed"]
    payload = {
        "status": "passed" if len(done) == len(DEEP_SITES) else "partial",
        "population": "one evaluation group per loaded deep site; not a 500-group intervention study",
        "post_api_jobs": "closed_503",
        "online_custom_compute": "not_opened",
        "pairs": pairs,
        "reason": f"{len(done)}/{len(DEEP_SITES)} deep sites have a real clean/edited pair. Unloaded sites were not filled from another model.",
        "model_sae_site": [row["site_id"] for row in done],
        "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume",
        "resources": "one resident model at a time; no quantization",
        "scientific_checks": "identity, ablate, scale, clamp, add, coalition, full reconstruction; single token position",
    }
    write_json(root / "GPU-04" / "decision.json", payload)
    write_json(root / "GPU-04" / "interventions" / "pairs.json", pairs)
    log(root, f"GPU-04 pairs={len(done)}")
    return payload


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _rss_occurrences(root: Path) -> int:
    """Each recorded setting repeats the clean capture plus 15 variants."""
    total = 0
    folder = root / "GPU-06" / "rss"
    if not folder.is_dir():
        return 0
    for path in sorted(folder.glob("*.jsonl")):
        if path.name.endswith("_invalid.jsonl"):
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                settings = json.loads(line).get("settings") or []
                total += 16 * len(settings)
    return total


def _rss_measured_ids(path: Path) -> set[str]:
    found = set()
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("settings"):
                found.add(row["sample_id"])
    return found


def _finished_ids(path: Path) -> set[str]:
    found = set()
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                found.add(json.loads(line)["sample_id"])
    return found


def _sar_main_tally(path: Path) -> dict[str, Any]:
    """Count non-rotation rows. A failed sample id is not a passed group."""
    counts = {"passed": 0, "failed": 0, "undefined": 0}
    error = None
    if not path.is_file():
        return {"counts": counts, "error": error}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("rotation"):
                continue
            status = row.get("status") or "failed"
            counts[status] = counts.get(status, 0) + 1
            if error is None and row.get("error"):
                error = str(row["error"])[:240]
    return {"counts": counts, "error": error}


def _sar_one(adapter, sae, site, tokens, paper) -> dict:
    from saebase.integrations.sar import evaluate_sar

    active = None
    snapshot_codes = None
    try:
        from saebase.api.features import capture_features

        snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
        snapshot_codes = snapshot.codes
        active = int(torch.count_nonzero(snapshot.codes).item())
    except Exception as exc:
        return {"status": "failed", "error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:240]}
    if active < K:
        return {"status": "undefined", "actual_k": active, "reason": "fewer than 16 active features; not labeled as K16"}
    readout = _readout_factory(adapter, tokens, site)
    result = evaluate_sar(
        adapter,
        sae,
        site=site,
        paper=paper,
        readout=readout,
        tokens=tokens,
        K=K,
        M=M,
        n_fit=N_FIT,
        seed=SEED,
        p=0.5,
        readout_position=-1,
        verify_replay=True,
    )
    metrics = {}
    for key, value in result.metrics.items():
        metrics[key] = None if value != value else value
    return {
        "status": "passed",
        "actual_k": K,
        "metrics": metrics,
        "feature_ids": [int(item) for item in result.feature_ids.tolist()],
        "g_full": readout.g_full,
        "n_responses": int(result.responses.shape[0]),
        "fit_count": int(result.fit_indices.shape[0]),
        "held_count": int(result.held_indices.shape[0]),
        "responses_sha256": __import__("hashlib").sha256(result.responses.tobytes()).hexdigest(),
        "undefined_metrics": [key for key, value in metrics.items() if value is None],
        "codes_l0": int(torch.count_nonzero(snapshot_codes).item()),
    }


def _readout_factory(adapter, tokens, site):
    baseline = adapter.forward_logits(tokens=tokens, readout_position=site.position)
    target = int(torch.argmax(baseline).item())

    def readout(logits: torch.Tensor) -> float:
        return float(logits[target].item())

    readout.target_token_id = target  # type: ignore[attr-defined]
    readout.g_full = float(baseline[target].item())  # type: ignore[attr-defined]
    return readout


def run_gpu05(root: Path, sources: dict) -> dict:
    import gc

    from run_gpu import _exact_tokens
    from saebase.api import ActivationSite, ModelAdapter

    _coverage_ok()
    paper = _paper(sources["sar_py"])
    passed = _gpu02_passed(root)
    records = _records(root)
    by_dataset: dict[str, list] = {}
    for record in records:
        by_dataset.setdefault(record["dataset_id"], []).append(record)
    rotation_ids = set()
    for rows in by_dataset.values():
        for record in sorted(rows, key=lambda row: row["sample_id"])[:5]:
            rotation_ids.add(record["sample_id"])
    site_rows = []
    total_main = 0
    total_rotation = 0
    for site_id in SAR_SITES:
        out = root / "GPU-05" / "sar" / f"{site_id.replace(':', '_')}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        if site_id not in passed:
            site_rows.append({"site_id": site_id, "status": "not_started", "reason": "base capture is absent; pilot numbers were not copied here"})
            continue
        site = _site(site_id)
        done = _finished_ids(out)
        if records and all(record["sample_id"] in done for record in records):
            tally = _sar_main_tally(out)
            counts = tally["counts"]
            site_rows.append(
                {
                    "site_id": site_id,
                    "status": "passed" if counts.get("passed") == 500 and counts.get("failed", 0) == 0 else "partial",
                    "measured_main_groups": counts.get("passed", 0),
                    "undefined_groups": counts.get("undefined", 0),
                    "failed_groups": counts.get("failed", 0),
                    "hook": site["hook"],
                    "error": tally["error"],
                }
            )
            continue
        log(root, f"GPU-05 {site_id}")
        try:
            model, sae, _meta = _load_pair(site)
        except Exception as exc:
            site_rows.append({"site_id": site_id, "status": "blocked", "reason": type(exc).__name__ + ": " + str(exc).splitlines()[0][:240]})
            continue
        adapter = ModelAdapter(model)
        activation_site = ActivationSite(int(site["layer"]), site["hook"], position=-1)
        failures = 0
        undefined = 0
        measured = 0
        try:
            masks = None
            with out.open("a", encoding="utf-8") as stream:
                for index, record in enumerate(records):
                    if record["sample_id"] in done:
                        continue
                    tokens, error = _exact_tokens(model, record["prompt"])
                    if error or tokens is None:
                        failures += 1
                        stream.write(json.dumps({"sample_id": record["sample_id"], "status": "failed", "error": error}) + "\n")
                        continue
                    try:
                        row = _sar_one(adapter, sae, activation_site, tokens, paper)
                    except Exception as exc:
                        message = type(exc).__name__ + ": " + str(exc).splitlines()[0][:240]
                        if "identity patch" in message.lower() or "Identity patch" in str(exc):
                            stream.write(json.dumps({"sample_id": record["sample_id"], "status": "failed", "error": message}) + "\n")
                            failures += 1
                            log(root, f"GPU-05 {site_id} stopped on identity patch")
                            break
                        failures += 1
                        stream.write(json.dumps({"sample_id": record["sample_id"], "status": "failed", "error": message}) + "\n")
                        continue
                    row["sample_id"] = record["sample_id"]
                    row["prompt_group_id"] = record["prompt_group_id"]
                    row["dataset_id"] = record["dataset_id"]
                    row["usage_split"] = record["usage_split"]
                    row["rotation"] = False
                    if row["status"] == "undefined":
                        undefined += 1
                    else:
                        measured += 1
                        total_main += 1
                    if masks is None and row["status"] == "passed":
                        masks = {"K": K, "M": M, "n_fit": N_FIT, "seed": SEED, "note": "same coalition design for every prompt; per-prompt feature ids differ"}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if record["sample_id"] in rotation_ids and row["status"] == "passed":
                        shifted = _rotation_control(adapter, sae, activation_site, tokens, paper)
                        shifted["sample_id"] = record["sample_id"]
                        shifted["rotation"] = True
                        shifted["rotation_definition"] = "same masks and seed; K selected feature ids shifted by one position"
                        stream.write(json.dumps(shifted, allow_nan=False) + "\n")
                        if shifted.get("status") == "passed":
                            total_rotation += 1
                    if index % 25 == 0:
                        log(root, f"GPU-05 {site_id} index={index} measured={measured}")
            if masks is not None:
                write_json(root / "GPU-05" / "sar" / f"{site_id.replace(':', '_')}_design.json", masks)
        finally:
            del model, sae, adapter
            gc.collect()
            torch.cuda.empty_cache()
        tally = _sar_main_tally(out)
        counts = tally["counts"]
        site_rows.append(
            {
                "site_id": site_id,
                "status": "passed" if counts.get("passed") == 500 and counts.get("failed", 0) == 0 else "partial",
                "measured_main_groups": counts.get("passed", 0),
                "undefined_groups": counts.get("undefined", 0),
                "failed_groups": counts.get("failed", 0),
                "hook": site["hook"],
                "error": tally["error"],
            }
        )
    total_main = 0
    total_rotation = 0
    for path in (root / "GPU-05" / "sar").glob("*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "passed":
                    continue
                if row.get("rotation"):
                    total_rotation += 1
                else:
                    total_main += 1
    planned = 3 * 500 * M + 50 * 3 * M
    decoder_error = next((row.get("error") for row in site_rows if row.get("error") and "not additive" in row["error"]), None)
    if decoder_error:
        reason = (
            "SAE decoder is not additive, so coalition groups measured stay 0. "
            "Rows with fewer than 16 active features stay undefined and are not labeled K16. "
            "Loaded sites are not marked passed."
        )
    else:
        reason = "Counts above are real coalition groups. Sites that did not load stay not_started and are not marked passed from the Qwen run."
    payload = {
        "status": "passed" if total_main == 1500 and total_rotation == 150 else "partial",
        "protocol": "K=16,M=384,fit=256,held-out=128",
        "planned_coalitions": planned,
        "main_groups_measured": total_main,
        "rotation_groups_measured": total_rotation,
        "logical_coalitions_measured": total_main * M + total_rotation * M,
        "sites": site_rows,
        "batch_parity": "sequential evaluate_sar; a batched patch is not substituted unless logit parity passes. This run uses the original sequential patch.",
        "statistical_unit": "base group",
        "reason": reason,
        "model_sae_site": [row["site_id"] for row in site_rows if row.get("status") != "not_started"],
        "population": {"main_groups": total_main, "rotation_groups": total_rotation, "planned": planned},
        "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume",
        "resources": "sequential full-prefix patch; no quantization",
        "scientific_checks": "original paper.coalitions and paper.sar; K stays 16; short support is undefined rather than relabeled",
    }
    write_json(root / "GPU-05" / "decision.json", payload)
    log(root, f"GPU-05 main_groups={total_main} rotation_groups={total_rotation}")
    return payload


def _rotation_control(adapter, sae, site, tokens, paper) -> dict:
    """Same coalition design, with the selected feature order rotated by one."""
    from saebase.api.features import capture_features, edit_features

    snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
    selected = torch.argsort(snapshot.codes.abs(), descending=True, stable=True)[:K].tolist()
    if len(selected) < K:
        return {"status": "undefined", "actual_k": len(selected)}
    rotated = selected[1:] + selected[:1]
    masks, fit, held = paper.coalitions(K=K, M=M, p=0.5, n_fit=N_FIT, seed=SEED)
    masks = np.asarray(masks, dtype=bool)
    baseline = adapter.forward_logits(tokens=tokens, readout_position=site.position)
    target = int(torch.argmax(baseline).item())
    g_full = float(baseline[target].item())
    responses = np.empty(M, dtype=np.float64)
    for index, mask in enumerate(masks):
        replacements = {int(fid): 0.0 for fid, keep in zip(rotated, mask) if not keep}
        if replacements:
            patch = edit_features(sae, snapshot, replacements=replacements)
            logits = adapter.patch_logits(tokens=tokens, site=site, activation=patch.activation, readout_position=-1)
        else:
            logits = baseline
        responses[index] = float(logits[target].item())
    metrics = paper.sar(responses, masks, np.asarray(fit), np.asarray(held), g_full=g_full)
    clean = {}
    for key, value in metrics.items():
        value = float(np.asarray(value))
        clean[str(key)] = None if value != value else value
    return {"status": "passed", "metrics": clean, "feature_ids": [int(item) for item in rotated], "matched_to_seed": SEED}


def _anchor_token(model, base_text: str, variant_text: str) -> int | None:
    """Last tokenizer index of the preserved anchor, or None if a token cuts it."""
    from backend.readouts import input_anchor

    tokenizer = getattr(model, "tokenizer", None)
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if inner is None:
        return None
    try:
        encoded = inner(variant_text, return_offsets_mapping=True, add_special_tokens=False)
        mapping = encoded["offset_mapping"]
        if mapping and isinstance(mapping[0][0], (list, tuple)):
            mapping = mapping[0]
        offsets = [(int(pair[0]), int(pair[1])) for pair in mapping]
        located = input_anchor(base_text, variant_text, anchor=base_text, offset_mapping=offsets)
    except Exception:
        return None
    positions = located.get("token_positions") or []
    if not positions:
        return None
    return int(positions[-1])


def run_gpu06(root: Path, sources: dict) -> dict:
    import gc

    from backend.core import read_json
    from backend.transforms import build_variants
    from run_gpu import _exact_tokens
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.integrations.rss import collect_rss_features, evaluate_rss

    _coverage_ok()
    paper = _paper(sources["rss_py"])
    passed = _gpu02_passed(root)
    lock = read_json(REPO / "apps/research-lab/data/source_lock_v1.json")
    revisions = {row["dataset_id"]: row["source_revision"] for row in lock["sources"]}
    records = _records(root)
    site_rows = []
    occurrences = 0
    for site_id in DEEP_SITES:
        out = root / "GPU-06" / "rss" / f"{site_id.replace(':', '_')}.jsonl"
        invalid_path = root / "GPU-06" / "rss" / f"{site_id.replace(':', '_')}_invalid.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        if site_id not in passed:
            site_rows.append({"site_id": site_id, "status": "not_started", "reason": "base capture is absent"})
            continue
        site = _site(site_id)
        done = _rss_measured_ids(out)
        if records and done and all(record["sample_id"] in done for record in records):
            invalid_count = _jsonl_count(invalid_path)
            site_rows.append(
                {
                    "site_id": site_id,
                    "status": "partial",
                    "groups_with_a_setting": len(done),
                    "groups_with_invalid_variant": invalid_count,
                }
            )
            continue
        log(root, f"GPU-06 {site_id}")
        try:
            model, sae, _meta = _load_pair(site)
        except Exception as exc:
            site_rows.append({"site_id": site_id, "status": "blocked", "reason": type(exc).__name__ + ": " + str(exc).splitlines()[0][:240]})
            continue
        adapter = ModelAdapter(model)
        activation_site = ActivationSite(int(site["layer"]), site["hook"], position=-1)
        if out.is_file() and not done:
            out.unlink()
            invalid_path.unlink(missing_ok=True)
        measured = 0
        invalid = 0
        try:
            with out.open("a", encoding="utf-8") as stream, invalid_path.open("a", encoding="utf-8") as invalid_stream:
                for index, record in enumerate(records):
                    if record["sample_id"] in done:
                        measured += 1
                        continue
                    if record["dataset_id"] == "humaneval":
                        pass  # comments and whitespace only; the variant is never executed
                    built = build_variants(
                        record["dataset_id"],
                        record["sample_id"],
                        record["prompt_group_id"],
                        record["prompt"],
                        anchor=record["prompt"],
                        source_revision=revisions[record["dataset_id"]],
                    )
                    settings = []
                    group_invalid = []
                    for setting in ("local_format", "instruction_wrapper", "context_wrapper"):
                        variants = [row for row in built["variants"] if row["setting"] == setting]
                        kept_selection = []
                        kept_held = []
                        for variant in variants:
                            position = _anchor_token(model, record["prompt"], variant["text"])
                            role_ok = (variant["variant_role"] == "selection" and variant["realization"] == 0) or (
                                variant["variant_role"] == "held_out" and variant["realization"] != 0
                            )
                            if position is None or not role_ok:
                                group_invalid.append({"variant_id": variant["variant_id"], "tokenizer_alignment": position is not None, "role_ok": role_ok})
                                continue
                            bucket = kept_selection if variant["variant_role"] == "selection" else kept_held
                            bucket.append((variant["text"], position))
                        if len(kept_selection) != 5 or len(kept_held) != 10:
                            group_invalid.append({"setting": setting, "selection": len(kept_selection), "held_out": len(kept_held)})
                            continue
                        clean_position = _anchor_token(model, record["prompt"], record["prompt"])
                        _tokens, error = _exact_tokens(model, record["prompt"])
                        if error or clean_position is None:
                            group_invalid.append({"setting": setting, "error": error or "clean_anchor_not_aligned"})
                            continue
                        features = collect_rss_features(
                            adapter,
                            sae,
                            site=ActivationSite(activation_site.layer, activation_site.hook_name, position=clean_position),
                            clean_prompt=record["prompt"],
                            selection_variants=[item[0] for item in kept_selection],
                            held_out_variants=[item[0] for item in kept_held],
                            selection_positions=[item[1] for item in kept_selection],
                            held_out_positions=[item[1] for item in kept_held],
                        )
                        evaluated = evaluate_rss(features, paper=paper)
                        support = [int(item) for item in evaluated.support.tolist()]
                        mean = evaluated.mean_rss
                        settings.append(
                            {
                                "setting": setting,
                                "support": support,
                                "support_status": "undefined" if len(support) == 0 else "measured",
                                "mean_rss": None if mean != mean else mean,
                                "held_out_rate_mean": None
                                if evaluated.held_out_rates.size == 0 or not np.isfinite(evaluated.held_out_rates).all()
                                else float(np.nanmean(evaluated.held_out_rates)),
                                "semantic_review": False,
                                "meaning_preserving": False if record["dataset_id"] in {"wikitext103", "tinystories"} else None,
                            }
                        )
                        occurrences += 1 + 15
                    if group_invalid:
                        invalid += 1
                        invalid_stream.write(
                            json.dumps({"sample_id": record["sample_id"], "invalid": group_invalid[:8]}, ensure_ascii=False) + "\n"
                        )
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": record["sample_id"],
                                "prompt_group_id": record["prompt_group_id"],
                                "dataset_id": record["dataset_id"],
                                "usage_split": record["usage_split"],
                                "settings": settings,
                                "semantic_review": False,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    if settings:
                        measured += 1
                    if index % 25 == 0:
                        log(root, f"GPU-06 {site_id} index={index}")
        finally:
            del model, sae, adapter
            gc.collect()
            torch.cuda.empty_cache()
        site_rows.append({"site_id": site_id, "status": "partial" if measured else "failed", "groups_with_a_setting": measured, "groups_with_invalid_variant": invalid})
    occurrences = _rss_occurrences(root)
    payload = {
        "status": "partial",
        "sites": site_rows,
        "site_input_occurrences_recorded": occurrences,
        "planned_max_occurrences": 6 * 500 * 46,
        "semantic_review": False,
        "selection_and_held_out_not_swapped": True,
        "reason": "Tokenizer-aligned settings were measured. semantic_review stays false. WikiText and TinyStories are not marked meaning-preserving. HumanEval text was not executed.",
        "model_sae_site": [row["site_id"] for row in site_rows if row.get("groups_with_a_setting")],
        "population": {"occurrences": occurrences, "planned": 6 * 500 * 46},
        "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume",
        "resources": "hook capture; full-prefix; one model resident",
        "scientific_checks": "original rss() and held_out_rate(); empty support is undefined",
    }
    write_json(root / "GPU-06" / "decision.json", payload)
    log(root, f"GPU-06 occurrences={occurrences}")
    return payload


def run_gpu07_support(root: Path) -> dict:
    """Drop all-zero columns only when the synthetic curve agrees."""
    from saebase.integrations.localization import LocalizationGrid, evaluate_localization

    status_path = root / "GPU-07" / "localize_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    codes = json.loads((root / "GPU-07" / "synthetic_reference_codes.json").read_text(encoding="utf-8"))
    synthetic = json.loads((root / "GPU-07" / "synthetic_known_factor_control.json").read_text(encoding="utf-8"))
    base = np.asarray(codes["feature_values"], dtype=np.float64)
    padded = np.concatenate([base, np.zeros((base.shape[0], 3), dtype=np.float64)], axis=1)
    ids = tuple(row["sample_id"] for row in synthetic["states"])
    full = evaluate_localization(LocalizationGrid(codes=padded, n_factors=6, state_sample_ids=ids, source="synthetic_logic_not_real_sae"))
    support_index = np.nonzero(np.any(padded != 0, axis=0))[0]
    reduced = evaluate_localization(
        LocalizationGrid(codes=padded[:, support_index], n_factors=6, state_sample_ids=ids, source="synthetic_logic_not_real_sae")
    )
    assignment_ok = np.array_equal(full.assignment[support_index], reduced.assignment)
    budget_ok = int(full.budget) == int(reduced.budget)
    parity = {
        "zero_columns_dropped": 3,
        "budget_full": int(full.budget),
        "budget_support": int(reduced.budget),
        "budget_match": budget_ok,
        "assignment_match_on_nonzero_columns": bool(assignment_ok),
        "not_a_learned_sae_result": True,
    }
    status["synthetic_zero_column_parity"] = parity
    if not (budget_ok and assignment_ok):
        status["real_model_curve_support"] = {
            "status": "partial",
            "reason": "Synthetic full-column and support-column curves disagree, so the real dictionary was not reduced.",
        }
        write_json(status_path, status)
        log(root, "GPU-07 support parity failed; real curve stays partial")
        return status
    private = root / "private" / "pythia410m_pythia410m_mlp_topk_L12_codes.jsonl"
    by_id: dict[str, list] = {}
    with private.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("dataset_id") == "controlled6":
                by_id[row["sample_id"]] = row["nonzero_codes"]
    real = json.loads((root / "GPU-07" / "real_model_six_factor_design.json").read_text(encoding="utf-8"))
    columns = sorted({int(index) for pairs in by_id.values() for index, _value in pairs})
    ordered = []
    for state in real["states"]:
        vector = np.zeros(len(columns), dtype=np.float64)
        location = {column: position for position, column in enumerate(columns)}
        for index, value in by_id[state["sample_id"]]:
            vector[location[int(index)]] = value
        ordered.append(vector)
    grid = LocalizationGrid(
        codes=np.vstack(ordered),
        n_factors=6,
        state_sample_ids=tuple(state["sample_id"] for state in real["states"]),
        source="real_checkpoint_nonzero_support_only",
    )
    result = evaluate_localization(grid)
    status["real_model_curve_support"] = {
        "status": "passed",
        "source_type": "real_checkpoint",
        "site_id": "pythia410m:pythia410m_mlp_topk:L12",
        "n_features_full_dictionary_not_materialized": True,
        "n_features_in_support": len(columns),
        "budget_at_90pct": int(result.budget),
        "total_variance": float(result.total_variance),
        "zero_columns_removed_after_synthetic_parity": True,
        "states_are_one_controlled_population": True,
    }
    write_json(status_path, status)
    log(root, f"GPU-07 support curve features={len(columns)} budget={int(result.budget)}")
    return status
