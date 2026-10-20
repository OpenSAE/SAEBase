"""Section 5.2 measurements on the current release.

Estimators stay in the original paper files. New rows never overwrite the
already accepted captures, the old SAR jsonl, the 99,200 RSS count, or
feature_store_v3.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import REPO, file_sha256, free_gib, gpu_is_idle, gpu_snapshot, log, write_json

sys.path.insert(0, str(REPO / "apps" / "research-lab"))

PROTOCOL = "decoder_mediated_residual_preserving_v1"
RSS_NAME = "RSS under proposed transformations (not verified meaning-preserving)"
PUBLIC_TEXT = "text unavailable for public release"
SCHEDULE = ((16, 384, 256, 128), (8, 128, 96, 32), (4, 64, 48, 16), (2, 32, 24, 8))
NEW_IDENTITY = {
    "pythia410m:pythia410m_mlp_topk:L4",
    "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L7",
    "r1_distill_qwen15b:r1_distill_qwen15b_resid_l20:L20",
    "gemma2_2b_it:gemma2_2b_it_matryoshka_l12_32k:L12",
}


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cohort(active: int) -> tuple[int, int, int, int] | None:
    for item in SCHEDULE:
        if active >= item[0]:
            return item
    return None


def _finite_metrics(metrics: dict) -> dict[str, float | None]:
    clean: dict[str, float | None] = {}
    for key, value in metrics.items():
        number = np.asarray(value, dtype=np.float64)
        if number.ndim != 0 or not np.isfinite(number):
            clean[str(key)] = None
        else:
            clean[str(key)] = float(number)
    return clean


def _residual(name: str, actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - reference.float()).abs()
    denom = max(float(reference.float().norm().item()), 1e-12)
    return {
        "edit": name,
        "abs_max": float(delta.max().item()),
        "abs_mean": float(delta.mean().item()),
        "rel_l2": float(delta.norm().item() / denom),
        "dtype": str(actual.dtype).replace("torch.", ""),
    }


def _diagnostic(sae: Any, snapshot: Any, indices: list[int], adapter: Any, tokens: torch.Tensor, site: Any) -> dict:
    from saebase.api.features import edit_features

    hidden = snapshot.activation
    singles = []
    rows = []
    for index in indices[:2]:
        edited = edit_features(sae, snapshot, replacements={index: 0.0}).activation
        delta = edited - hidden
        singles.append(delta)
        rows.append(_residual(f"single:{index}", edited, hidden + delta))
    if len(indices) >= 2:
        pair = indices[:2]
        actual = edit_features(sae, snapshot, replacements={i: 0.0 for i in pair}).activation
        additive = hidden + singles[0] + singles[1]
        rounded = hidden.to(torch.bfloat16) + singles[0].to(torch.bfloat16) + singles[1].to(torch.bfloat16)
        rows.append(_residual("pair", actual, additive))
        rows.append(_residual("pair_bf16_reference", actual, rounded.float()))
        full = edit_features(sae, snapshot, replacements={i: 0.0 for i in indices}).activation
        additive_full = hidden + sum(singles, torch.zeros_like(hidden))
        if len(indices) > 2:
            extra = [
                edit_features(sae, snapshot, replacements={i: 0.0}).activation - hidden
                for i in indices[2:]
            ]
            additive_full = hidden + sum([*singles, *extra], torch.zeros_like(hidden))
        rows.append(_residual("all_selected", full, additive_full))
    baseline = adapter.forward_logits(tokens=tokens, readout_position=-1)
    replay = adapter.patch_logits(tokens=tokens, site=site, activation=hidden, readout_position=-1)
    replay_again = adapter.patch_logits(tokens=tokens, site=site, activation=hidden, readout_position=-1)
    identity_gap = float((baseline.float() - replay.float()).abs().max().item())
    repeat_gap = float((replay.float() - replay_again.float()).abs().max().item())
    return {
        "feature_ids": [int(item) for item in indices],
        "residuals": rows,
        "identity_patch_abs_max": identity_gap,
        "identity_patch_repeat_abs_max": repeat_gap,
        "activation_dtype": str(hidden.dtype).replace("torch.", ""),
    }


def _position(length: int, position: int) -> int:
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError(f"position outside length {length}")
    return position


def _batch_scalars(adapter: Any, tokens: torch.Tensor, site: Any, activations: list[torch.Tensor], target: int) -> list[float]:
    model = adapter.model
    if not callable(getattr(model, "run_with_hooks", None)):
        raise RuntimeError("batch path requires run_with_hooks")
    batch = len(activations)
    ids = tokens.repeat(batch, 1)
    position = _position(int(tokens.shape[1]), site.position)
    readout = _position(int(tokens.shape[1]), -1)
    stacked = torch.stack([item.detach() for item in activations])

    def replace(hidden: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        if hidden.ndim != 3 or hidden.shape[0] != batch or hidden.shape[1] != ids.shape[1]:
            raise RuntimeError("batched hook activation does not match the repeated tokens")
        patched = hidden.clone()
        patched[torch.arange(batch, device=hidden.device), position] = stacked.to(device=hidden.device, dtype=hidden.dtype)
        return patched

    output = model.run_with_hooks(ids, fwd_hooks=[(site.hook_name, replace)])
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor) or logits.shape[0] != batch:
        raise RuntimeError("batched forward did not return one logit row per coalition")
    return [float(logits[index, readout, target].item()) for index in range(batch)]


def _responses(
    adapter: Any,
    sae: Any,
    snapshot: Any,
    site: Any,
    tokens: torch.Tensor,
    selected: list[int],
    masks: np.ndarray,
    target: int,
    *,
    allow_batch: bool,
) -> tuple[np.ndarray, dict]:
    from saebase.api.features import edit_features

    activations = []
    for mask in masks:
        replacements = {int(feature): 0.0 for feature, keep in zip(selected, mask.tolist()) if not keep}
        if replacements:
            activations.append(edit_features(sae, snapshot, replacements=replacements).activation)
        else:
            activations.append(snapshot.activation)
    sequential = []
    check = min(4, len(activations))
    for activation in activations[:check]:
        logits = adapter.patch_logits(tokens=tokens, site=site, activation=activation, readout_position=-1)
        sequential.append(float(logits[target].item()))
    note = {"mode": "sequential_full_prefix", "kv_cache": False, "parity": "not_used"}
    use_batch = False
    if allow_batch and check >= 2 and callable(getattr(adapter.model, "run_with_hooks", None)):
        try:
            batched = _batch_scalars(adapter, tokens, site, activations[:check], target)
            gaps = [abs(left - right) for left, right in zip(sequential, batched)]
            scale = max(1.0, max(abs(value) for value in sequential))
            matched = max(gaps) <= 1e-3 * scale
            note = {
                "mode": "batched_hook" if matched else "sequential_full_prefix",
                "kv_cache": False,
                "parity": "matched" if matched else "mismatch_fallback",
                "abs_max_gap": max(gaps),
                "checked": check,
            }
            use_batch = matched
        except Exception as exc:
            note = {
                "mode": "sequential_full_prefix",
                "kv_cache": False,
                "parity": "error_fallback",
                "error": type(exc).__name__,
            }
    values = np.empty(len(activations), dtype=np.float64)
    if use_batch:
        values[:check] = sequential
        cursor = check
        while cursor < len(activations):
            chunk = activations[cursor : cursor + 4]
            try:
                values[cursor : cursor + len(chunk)] = _batch_scalars(adapter, tokens, site, chunk, target)
            except RuntimeError:
                for offset, activation in enumerate(chunk):
                    logits = adapter.patch_logits(tokens=tokens, site=site, activation=activation, readout_position=-1)
                    values[cursor + offset] = float(logits[target].item())
                note["mode"] = "sequential_full_prefix"
                note["parity"] = "batch_runtime_fallback"
            cursor += len(chunk)
    else:
        values[:check] = sequential
        for index, activation in enumerate(activations[check:], start=check):
            logits = adapter.patch_logits(tokens=tokens, site=site, activation=activation, readout_position=-1)
            values[index] = float(logits[target].item())
    return values, note


def _sar_summary(root: Path, rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    failed = 0
    files = 0
    for path in (root / "GPU-05" / "sar_v2").glob("*/*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        files += 1
        if row.get("status") == "failed":
            failed += 1
        key = f"{row.get('status')}:K{row.get('k')}"
        counts[key] = counts.get(key, 0) + 1
    status = "passed" if files >= 1500 and failed == 0 and any(key.startswith("passed:") for key in counts) else "partial"
    payload = {
        "status": status,
        "protocol": PROTOCOL,
        "analysis": "SAR (decoder-mediated intervention protocol)",
        "rotation_status": "rotation_pending",
        "planned_rotation_measurements": 57600,
        "permutation_removed_from_main_denominator": True,
        "cohorts_not_pooled": True,
        "population": counts,
        "files": files,
        "sites": rows,
        "reason": "K16, K8, K4 and K2 stay in separate cohorts. Undefined is not relabeled as K16. Rotation stays rotation_pending. The old failed jsonl is kept.",
        "scientific_checks": "original paper.coalitions and paper.sar after edit_features and patch_logits; evaluate_sar still defaults require_affine_decode=True; two-prompt residuals are recorded before the batch",
        "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume",
        "resources": "one resident hook model; batch only after scalar parity; no KV cache",
        "model_sae_site": [row.get("site_id") for row in rows],
    }
    write_json(root / "GPU-05" / "decision_v2.json", payload)


def run_sar_v2(root: Path, sources: dict) -> dict:
    import gc

    from analyses import SAR_SITES, _coverage_ok, _gpu02_passed, _load_pair, _paper, _records, _site
    from run_gpu import _exact_tokens
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.api.features import capture_features
    from saebase.store.identity import sha256_state_dict

    _coverage_ok()
    if free_gib(root) < 40:
        raise RuntimeError("free disk is below 40 GiB; not starting SAR")
    paper = _paper(sources["sar_py"])
    estimator_sha = sources["sar_sha256"]
    passed = _gpu02_passed(root)
    records = _records(root)
    site_rows = []
    diagnostic_done = (root / "GPU-05" / "sar_v2" / "diagnostic.json").is_file()
    batch_ok: dict[str, bool] = {}
    for site_id in SAR_SITES:
        folder = root / "GPU-05" / "sar_v2" / _safe(site_id)
        folder.mkdir(parents=True, exist_ok=True)
        if site_id not in passed:
            site_rows.append({"site_id": site_id, "status": "blocked", "reason": "base capture is absent"})
            continue
        pending = [record for record in records if not (folder / f"{_safe(record['sample_id'])}.json").is_file()]
        existing = len(records) - len(pending)
        if not pending:
            site_rows.append({"site_id": site_id, "status": "measured", "files": existing})
            continue
        log(root, f"SAR v2 {site_id} pending={len(pending)} done={existing}")
        site = _site(site_id)
        model, sae, _meta = _load_pair(site)
        adapter = ModelAdapter(model)
        activation = ActivationSite(int(site["layer"]), site["hook"], position=-1)
        model_sha = sha256_state_dict(model)
        sae_sha = sha256_state_dict(sae)
        started = time.time()
        finished = 0
        try:
            for record in pending:
                tokens, error = _exact_tokens(model, record["prompt"])
                destination = folder / f"{_safe(record['sample_id'])}.json"
                if error or tokens is None:
                    _atomic(destination, {"sample_id": record["sample_id"], "status": "failed", "error": error or "tokens"})
                    continue
                snapshot = capture_features(adapter, sae, site=activation, tokens=tokens)
                active = int(torch.count_nonzero(snapshot.codes).item())
                chosen = _cohort(active)
                base = {
                    "sample_id": record["sample_id"],
                    "dataset_id": record["dataset_id"],
                    "protocol": PROTOCOL,
                    "model_sha256": model_sha,
                    "sae_sha256": sae_sha,
                    "estimator_sha256": estimator_sha,
                    "active_features": active,
                }
                if chosen is None:
                    _atomic(destination, {**base, "status": "undefined", "reason": "fewer than 2 editable features"})
                    continue
                k, m_count, n_fit, _held = chosen
                selected = torch.argsort(snapshot.codes.abs(), descending=True, stable=True)[:k].tolist()
                if not diagnostic_done:
                    diag = _diagnostic(sae, snapshot, selected, adapter, tokens, activation)
                    diag["sample_id"] = record["sample_id"]
                    diag["site_id"] = site_id
                    write_json(root / "GPU-05" / "sar_v2" / "diagnostic.json", {"prompts": [diag], "protocol": PROTOCOL})
                    diagnostic_done = True
                elif (root / "GPU-05" / "sar_v2" / "diagnostic.json").is_file():
                    current = _load(root / "GPU-05" / "sar_v2" / "diagnostic.json")
                    if len(current.get("prompts", [])) < 2 and current.get("prompts", [{}])[0].get("sample_id") != record["sample_id"]:
                        diag = _diagnostic(sae, snapshot, selected, adapter, tokens, activation)
                        diag["sample_id"] = record["sample_id"]
                        diag["site_id"] = site_id
                        current.setdefault("prompts", []).append(diag)
                        write_json(root / "GPU-05" / "sar_v2" / "diagnostic.json", current)
                baseline = adapter.forward_logits(tokens=tokens, readout_position=-1)
                replay = adapter.patch_logits(tokens=tokens, site=activation, activation=snapshot.activation, readout_position=-1)
                if not torch.allclose(baseline.float(), replay.float(), rtol=1e-4, atol=1e-4):
                    gap = float((baseline.float() - replay.float()).abs().max().item())
                    _atomic(destination, {**base, "status": "failed", "error": f"identity_patch_abs_max={gap}"})
                    continue
                target = int(torch.argmax(baseline).item())
                masks, fit, held = paper.coalitions(K=k, M=m_count, p=0.5, n_fit=n_fit, seed=20261018)
                masks = np.asarray(masks, dtype=bool)
                responses, accel = _responses(
                    adapter, sae, snapshot, activation, tokens, selected, masks, target, allow_batch=batch_ok.get(site_id, True)
                )
                batch_ok[site_id] = accel.get("mode") == "batched_hook"
                if float(np.nanstd(responses)) == 0.0:
                    _atomic(
                        destination,
                        {
                            **base,
                            "status": "undefined",
                            "reason": "response variance is zero",
                            "k": k,
                            "g": responses.tolist(),
                            "feature_ids": [int(item) for item in selected],
                        },
                    )
                    continue
                metrics = paper.sar(responses, masks, np.asarray(fit), np.asarray(held), g_full=float(baseline[target].item()))
                digest = _atomic(
                    destination,
                    {
                        **base,
                        "status": "passed",
                        "k": k,
                        "M": m_count,
                        "n_fit": n_fit,
                        "n_held": int(len(held)),
                        "feature_ids": [int(item) for item in selected],
                        "masks": masks.astype(int).tolist(),
                        "g": responses.tolist(),
                        "g_full": float(baseline[target].item()),
                        "metrics": _finite_metrics(metrics),
                        "acceleration": accel,
                        "diagnostic_recorded": True,
                    },
                )
                finished += 1
                if finished % 10 == 0:
                    rate = (time.time() - started) / finished
                    remaining = rate * (len(pending) - finished)
                    log(root, f"SAR v2 {site_id} done={existing + finished} sha256={digest[:12]} eta_s={remaining:.0f}")
        finally:
            del model, sae, adapter
            gc.collect()
            torch.cuda.empty_cache()
        site_rows.append({"site_id": site_id, "status": "measured", "files": len(list(folder.glob('*.json')))})
        _sar_summary(root, site_rows)
    _sar_summary(root, site_rows)
    return {"sites": site_rows}


def _token_ids(model: Any, text: str) -> list[int]:
    convert = model.to_tokens
    try:
        tokens = convert(text, truncate=False)
    except TypeError:
        tokens = convert(text)
    return [int(item) for item in tokens[0].detach().cpu().tolist()]


def _aligned(model: Any, base_text: str, variant_text: str) -> dict[str, Any]:
    from backend.readouts import input_anchor

    empty = {
        "offset_len": 0,
        "model_len": 0,
        "offsets": [],
        "bos_shift": None,
        "anchor_in_offset_sequence": None,
        "model_index": None,
        "model_token_ids": [],
        "offset_token_ids": [],
    }
    try:
        tokenizer = getattr(model, "tokenizer", None)
        inner = getattr(tokenizer, "tokenizer", tokenizer)
        encoded = inner(variant_text, return_offsets_mapping=True, add_special_tokens=False)
        ids = encoded["input_ids"]
        mapping = encoded["offset_mapping"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
            mapping = mapping[0]
        offsets = [[int(pair[0]), int(pair[1])] for pair in mapping]
        plain = [int(item) for item in ids]
        actual = _token_ids(model, variant_text)
        shift = None
        for candidate in range(0, min(8, len(actual) + 1)):
            if actual[candidate : candidate + len(plain)] == plain:
                shift = candidate
                break
        policy = "exact_token_span_v1"
        boundary_sensitive = False
        try:
            located = input_anchor(base_text, variant_text, anchor=base_text, offset_mapping=[tuple(item) for item in offsets])
            positions = located.get("token_positions") or []
            anchor = int(positions[-1]) if positions else None
            reason = None if anchor is not None else "anchor_not_located"
        except ValueError as exc:
            message = str(exc).splitlines()[0]
            anchor = None
            reason = "anchor_cuts_token" if "cuts through" in message else message[:160]
            if reason == "anchor_cuts_token" and variant_text.count(base_text) == 1:
                end = variant_text.index(base_text) + len(base_text)
                for index, (start, stop) in enumerate(offsets):
                    if start <= end - 1 < stop:
                        anchor = index
                        policy = "content_end_token_v1"
                        boundary_sensitive = start < variant_text.index(base_text) or stop > end
                        reason = None
                        break
        model_index = None if anchor is None or shift is None else anchor + shift
        if shift is None and reason is None:
            reason = "bos_shift_unmatched"
        if model_index is not None and not 0 <= model_index < len(actual):
            model_index = None
            reason = "model_index_outside_sequence"
        return {
            "offset_token_ids_sha256": hashlib.sha256(json.dumps(plain).encode("utf-8")).hexdigest(),
            "model_token_ids_sha256": hashlib.sha256(json.dumps(actual).encode("utf-8")).hexdigest(),
            "offset_len": len(plain),
            "model_len": len(actual),
            "offsets": offsets,
            "bos_shift": shift,
            "anchor_in_offset_sequence": anchor,
            "model_index": model_index,
            "anchor_policy": policy,
            "boundary_sensitive": boundary_sensitive,
            "reason": reason,
            "model_token_ids": actual,
            "offset_token_ids": plain,
        }
    except Exception as exc:
        empty["reason"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:160]
        return empty


def _setting_open(name: str, setting: dict | None) -> bool:
    if not setting:
        return True
    if setting.get("status") in {"measured", "undefined"}:
        return False
    return name == "local_format" and setting.get("anchor_policy") != "content_end_token_v1"


def _legacy_rss(root: Path) -> dict[str, int]:
    from analyses import _rss_occurrences

    return {"position_unverified_legacy_nominal_calls": _rss_occurrences(root)}


def run_rss_v2(root: Path, sources: dict) -> dict:
    import gc

    from analyses import DEEP_SITES, _coverage_ok, _gpu02_passed, _load_pair, _paper, _records, _site
    from backend.core import read_json
    from backend.transforms import build_variants
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.api.features import capture_features
    from saebase.integrations.rss import RSSFeatures, evaluate_rss

    _coverage_ok()
    paper = _paper(sources["rss_py"])
    passed = _gpu02_passed(root)
    lock = read_json(REPO / "apps/research-lab/data/source_lock_v1.json")
    revisions = {row["dataset_id"]: row["source_revision"] for row in lock["sources"]}
    records = _records(root)
    probe_path = root / "GPU-06" / "rss_v2" / "alignment_probe.json"
    site_rows = []
    for site_id in DEEP_SITES:
        folder = root / "GPU-06" / "rss_v2" / _safe(site_id)
        folder.mkdir(parents=True, exist_ok=True)
        if site_id not in passed:
            site_rows.append({"site_id": site_id, "status": "blocked"})
            continue
        if not any(
            _setting_open(name, (_load(folder / f"{_safe(record['sample_id'])}.json").get("settings") or {}).get(name))
            for record in records
            for name in ("local_format", "instruction_wrapper", "context_wrapper")
        ):
            site_rows.append({"site_id": site_id, "files": len(list(folder.glob("*.json"))), "status": "complete"})
            continue
        site = _site(site_id)
        model, sae, _meta = _load_pair(site)
        adapter = ModelAdapter(model)
        started = time.time()
        done = 0
        try:
            for record in records:
                destination = folder / f"{_safe(record['sample_id'])}.json"
                current = _load(destination)
                settings = current.get("settings") or {}
                missing = [name for name in ("local_format", "instruction_wrapper", "context_wrapper") if _setting_open(name, settings.get(name))]
                if not missing:
                    done += 1
                    continue
                built = build_variants(
                    record["dataset_id"],
                    record["sample_id"],
                    record["prompt_group_id"],
                    record["prompt"],
                    anchor=record["prompt"],
                    source_revision=revisions[record["dataset_id"]],
                )
                clean_align = _aligned(model, record["prompt"], record["prompt"])
                if not probe_path.is_file():
                    variant = next(row for row in built["variants"] if row["setting"] == "local_format")
                    wrapped = _aligned(model, record["prompt"], variant["text"])
                    write_json(
                        probe_path,
                        {
                            "clean": {k: v for k, v in clean_align.items() if k != "offsets"},
                            "wrapper": {
                                "setting": variant["setting"],
                                "variant_id": variant["variant_id"],
                                **{k: v for k, v in wrapped.items()},
                            },
                        },
                    )
                if clean_align["model_index"] is None:
                    for name in missing:
                        settings[name] = {"status": "invalid", "reason": "clean_anchor"}
                else:
                    clean_site = ActivationSite(int(site["layer"]), site["hook"], position=int(clean_align["model_index"]))
                    clean_ids = adapter.tokens(prompt=record["prompt"])
                    clean_codes = capture_features(adapter, sae, tokens=clean_ids, site=clean_site).codes.detach().cpu().numpy()
                    for name in missing:
                        variants = [row for row in built["variants"] if row["setting"] == name]
                        selection = []
                        held = []
                        invalid = []
                        for variant in variants:
                            role_ok = (variant["variant_role"] == "selection" and variant["realization"] == 0) or (
                                variant["variant_role"] == "held_out" and variant["realization"] != 0
                            )
                            aligned = _aligned(model, record["prompt"], variant["text"])
                            if not role_ok or aligned["model_index"] is None:
                                invalid.append({"variant_id": variant["variant_id"], "reason": aligned.get("reason") or "role_rejected"})
                                continue
                            bucket = selection if variant["variant_role"] == "selection" else held
                            bucket.append((variant, aligned))
                        if len(selection) != 5 or len(held) != 10:
                            settings[name] = {
                                "status": "invalid",
                                "reason": "fewer_than_5_selection_or_10_held_out",
                                "anchor_policy": "content_end_token_v1",
                                "selection": len(selection),
                                "held_out": len(held),
                                "invalid_variant_ids": [item["variant_id"] for item in invalid],
                                "invalid_reasons": invalid,
                            }
                            continue
                        selection_codes = []
                        held_codes = []
                        for variant, aligned in selection:
                            ids = adapter.tokens(prompt=variant["text"])
                            chosen = ActivationSite(clean_site.layer, clean_site.hook_name, position=int(aligned["model_index"]))
                            selection_codes.append(capture_features(adapter, sae, tokens=ids, site=chosen).codes.detach().cpu().numpy())
                        for variant, aligned in held:
                            ids = adapter.tokens(prompt=variant["text"])
                            chosen = ActivationSite(clean_site.layer, clean_site.hook_name, position=int(aligned["model_index"]))
                            held_codes.append(capture_features(adapter, sae, tokens=ids, site=chosen).codes.detach().cpu().numpy())
                        features = RSSFeatures(
                            clean_codes=clean_codes.astype(np.float64, copy=True),
                            selection_codes=np.stack(selection_codes).astype(np.float64),
                            held_out_codes=np.stack(held_codes).astype(np.float64),
                            selection_changes=np.stack(selection_codes).astype(np.float64) - clean_codes.astype(np.float64),
                            held_out_changes=np.stack(held_codes).astype(np.float64) - clean_codes.astype(np.float64),
                            selection_positions=tuple(int(item[1]["model_index"]) for item in selection),
                            held_out_positions=tuple(int(item[1]["model_index"]) for item in held),
                            clean_position=int(clean_align["model_index"]),
                            site=clean_site,
                        )
                        evaluated = evaluate_rss(features, paper=paper)
                        support = [int(item) for item in evaluated.support.tolist()]
                        mean = evaluated.mean_rss
                        settings[name] = {
                            "status": "undefined" if len(support) == 0 else "measured",
                            "support": support,
                            "mean_rss": None if mean != mean else float(mean),
                            "held_out_rate_mean": None
                            if evaluated.held_out_rates.size == 0 or not np.isfinite(evaluated.held_out_rates).all()
                            else float(np.nanmean(evaluated.held_out_rates)),
                            "selection_variant_ids": [item[0]["variant_id"] for item in selection],
                            "held_out_variant_ids": [item[0]["variant_id"] for item in held],
                            "invalid_variant_ids": invalid,
                            "position_recomputed": True,
                            "semantic_review": False,
                            "meaning_preserving": False,
                            "anchor_policy": "content_end_token_v1",
                        }
                digest = _atomic(
                    destination,
                    {
                        "sample_id": record["sample_id"],
                        "dataset_id": record["dataset_id"],
                        "name": RSS_NAME,
                        "semantic_review": False,
                        "position_recomputed": True,
                        "settings": settings,
                    },
                )
                done += 1
                if done % 10 == 0:
                    rate = max((time.time() - started) / max(done, 1), 0.001)
                    log(root, f"RSS v2 {site_id} done={done} sha256={digest[:12]} eta_s={rate * (len(records) - done):.0f}")
        finally:
            del model, sae, adapter
            gc.collect()
            torch.cuda.empty_cache()
        site_rows.append({"site_id": site_id, "files": len(list(folder.glob('*.json')))})
    coverage = _coverage(root, site_rows)
    write_json(root / "GPU-06" / "coverage_v2.json", coverage)
    write_json(
        root / "GPU-06" / "decision_v2.json",
        {
            "status": "passed" if int(coverage.get("n_parent_groups") or 0) >= 3000 else "partial",
            "name": RSS_NAME,
            "coverage": coverage,
            "population": coverage,
            "sites": site_rows,
            "model_sae_site": [row["site_id"] for row in site_rows],
            "reason": "Old 99200 nominal calls stay position_unverified_legacy. New rows use recomputed positions and are not labeled meaning-preserving. semantic_review stays false.",
            "scientific_checks": "original rss() and held_out_rate(); one finished setting does not block the other two; selection and held-out were not swapped",
            "commands": "python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume",
            "resources": "one resident hook model; clean capture shared across the three settings",
        },
    )
    return coverage


def _coverage(root: Path, site_rows: list[dict]) -> dict[str, Any]:
    legacy = _legacy_rss(root)
    groups = 0
    complete = 0
    selection = 0
    held = 0
    invalid = 0
    unique = set()
    for path in (root / "GPU-06" / "rss_v2").glob("*/*.json"):
        row = _load(path)
        groups += 1
        settings = row.get("settings") or {}
        if all(settings.get(name, {}).get("status") in {"measured", "undefined"} for name in ("local_format", "instruction_wrapper", "context_wrapper")):
            complete += 1
        unique.add((path.parent.name, row.get("sample_id"), "clean"))
        for name, setting in settings.items():
            selection += len(setting.get("selection_variant_ids") or [])
            held += len(setting.get("held_out_variant_ids") or [])
            invalid += len(setting.get("invalid_variant_ids") or [])
            for variant in setting.get("selection_variant_ids") or []:
                unique.add((path.parent.name, row.get("sample_id"), variant))
            for variant in setting.get("held_out_variant_ids") or []:
                unique.add((path.parent.name, row.get("sample_id"), variant))
    legacy.update(
        {
            "n_parent_groups": groups,
            "settings_complete": complete,
            "selection_variants": selection,
            "heldout_variants": held,
            "unique_site_input_occurrences": len(unique),
            "invalid": invalid,
            "position_recomputed": True,
            "planned_unique_ceiling": 6 * 500 * 46,
            "sites": site_rows,
            "name": RSS_NAME,
            "semantic_review": False,
        }
    )
    return legacy


def _localize(root: Path) -> None:
    path = root / "GPU-07" / "localize_status.json"
    status = _load(path)
    support = status.get("real_model_curve_support") or {}
    if support.get("n_features_in_support") == 55:
        status["real_model_scope_on_observed_support"] = {
            **support,
            "name": "real_model_scope_on_observed_support",
            "synthetic_curve_name": "synthetic_logic_not_real_sae",
            "full_dictionary_not_allocated": True,
        }
        write_json(path, status)


def _identities(root: Path) -> None:
    decision = _load(root / "GPU-02" / "decision.json")
    rows = []
    for row in decision.get("site_results") or []:
        if row.get("site_id") in NEW_IDENTITY or row.get("hash_matches_historical") is False:
            rows.append(
                {
                    "site_id": row.get("site_id"),
                    "label": "new_run_identity",
                    "historical_comparison": "not_matched",
                    "loaded_model_sha256": row.get("loaded_model_sha256"),
                    "loaded_sae_sha256": row.get("loaded_sae_sha256"),
                    "measured": row.get("measured"),
                }
            )
    write_json(root / "GPU-02" / "new_run_identity.json", {"rows": rows, "inserted_into_old_hash_index": False})


def _cards(root: Path) -> dict:
    from collections import Counter

    from later import _codes

    cards = []
    by_site: dict[str, list] = {}
    for row in _codes(root):
        if row.get("site_id") in {
            "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14",
            "gemma1b:gemma1b_scope_resid:L13",
            "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22",
            "gemma3_4b_it:gemma3_4b_it_scope_resid:L22",
            "llama31_8b:llama31_8b_lxr_8x:L21",
            "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16",
        }:
            by_site.setdefault(row["site_id"], []).append(row)
    for site_id, rows in sorted(by_site.items()):
        counts: Counter[int] = Counter()
        peak: dict[int, tuple] = {}
        for row in rows:
            token_ids = row.get("token_ids") or []
            position = int(row.get("token_position", -1))
            selected = int(token_ids[position]) if 0 <= position < len(token_ids) else None
            for feature, value in row.get("nonzero_codes") or []:
                feature = int(feature)
                counts[feature] += 1
                previous = peak.get(feature)
                if previous is None or abs(float(value)) > previous[0]:
                    peak[feature] = (abs(float(value)), row["sample_id"], position, selected, row.get("token_ids_sha256"))
        for feature, count in counts.most_common(50):
            activation, sample_id, position, selected, token_sha = peak[feature]
            cards.append(
                {
                    "site_id": site_id,
                    "feature_id": feature,
                    "activation": activation,
                    "sample_id": sample_id,
                    "token_position": position,
                    "selected_token_id": selected,
                    "token_ids_sha256": token_sha,
                    "private_sample_reference": f"private codes {site_id} {sample_id}",
                    "clean_edited_reference": f"GPU-04 clean/edited pair for {site_id}; card is the clean capture",
                    "text": PUBLIC_TEXT,
                }
            )
    payload = {"card_count": len(cards), "target": 300, "sites": sorted(by_site), "cards": cards, "text": PUBLIC_TEXT}
    write_json(root / "GPU-03" / "deep_site_cards.json", payload)
    return payload


def _generation(root: Path) -> dict:
    import gc

    from accel import probe_vllm
    from analyses import _load_pair, _records, _site
    from run_gpu import _exact_tokens
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.api.features import capture_features, edit_features

    probe = probe_vllm(root)
    site_id = "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14"
    record = next(row for row in _records(root) if row["dataset_id"] != "humaneval")
    site = _site(site_id)
    model, sae, _meta = _load_pair(site)
    torch.manual_seed(20261018)
    adapter = ModelAdapter(model)
    tokens, error = _exact_tokens(model, record["prompt"])
    if error or tokens is None:
        payload = {"status": "failed", "error": error, "kv_cache": False}
        write_json(root / "GPU-04" / "generation_v2.json", payload)
        return payload
    activation = ActivationSite(int(site["layer"]), site["hook"], position=-1)
    try:
        baseline = adapter.forward_logits(tokens=tokens, readout_position=-1)
        snapshot = capture_features(adapter, sae, tokens=tokens, site=activation)
        feature = int(torch.argmax(snapshot.codes.abs()).item())
        edited = edit_features(sae, snapshot, replacements={feature: 0.0})
        patched = adapter.patch_logits(tokens=tokens, site=activation, activation=edited.activation, readout_position=-1)
        payload = {
            "status": "full_steered_generation_pending",
            "site_id": site_id,
            "sample_id": record["sample_id"],
            "seed": 20261018,
            "feature_id": feature,
            "baseline_top5": [int(item) for item in torch.topk(baseline, 5).indices.tolist()],
            "edited_top5": [int(item) for item in torch.topk(patched, 5).indices.tolist()],
            "kv_cache": False,
            "vllm": probe.get("decision"),
            "text": PUBLIC_TEXT,
            "pairs_not_rerun": True,
        }
    finally:
        del model, sae, adapter
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root / "GPU-04" / "generation_v2.json", payload)
    write_json(
        root / "GPU-08" / "new_observations.json",
        {"store": None, "reason": "no new feature rows; feature_store_v3 was not overwritten", "v3_preserved": True},
    )
    return payload


def _lab(root: Path) -> dict:
    import os
    import subprocess
    import urllib.error
    import urllib.request

    index = root / "lab_index"
    cards = _load(root / "GPU-03" / "deep_site_cards.json").get("cards") or []
    features = [
        {
            "feature_id": card["feature_id"],
            "site_id": card["site_id"],
            "label": PUBLIC_TEXT,
            "model": card["site_id"].split(":", 1)[0],
        }
        for card in cards[:100]
    ]
    analyses = [
        {"kind": "sar_v2", "protocol": PROTOCOL, "release": root.name, "text": PUBLIC_TEXT},
        {"kind": "rss_v2", "protocol": RSS_NAME, "release": root.name, "text": PUBLIC_TEXT},
    ]
    report_id = hashlib.sha256(root.name.encode("utf-8")).hexdigest()[:40]
    report = {
        "report_id": report_id,
        "title": "Work Plan 05 release index",
        "kind": "release",
        "visibility": "public",
        "source": root.name,
        "summary": PUBLIC_TEXT,
        "model": "redacted",
    }
    _publish(index / "features", "opensae.feature_index.v1", features)
    _publish(index / "analyses", "opensae.analysis_index.v1", analyses)
    report_dir = index / "reports" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    write_json(report_dir / "manifest.json", {"schema": "opensae.public_report.v1", "report_sha256": file_sha256(report_path)})
    env = os.environ.copy()
    env["OPENSAE_LAB_DATA"] = str(index)
    err_path = root / "GPU-09" / "local_lab_stderr.log"
    err_path.parent.mkdir(parents=True, exist_ok=True)
    err_stream = err_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["/venv/main/bin/python", "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd=str(REPO / "apps" / "research-lab"),
        env=env,
        stdout=err_stream,
        stderr=subprocess.STDOUT,
    )
    checks = {}
    ready = False
    try:
        for _ in range(60):
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=1)
                ready = True
                break
            except Exception:
                time.sleep(0.5)
        if not ready:
            err_stream.flush()
            raise RuntimeError("Research Lab did not accept connections: " + err_path.read_text(encoding="utf-8")[-500:])
        for path in ("/api/results", "/api/features", "/api/analyses", "/api/reports"):
            with urllib.request.urlopen("http://127.0.0.1:8765" + path, timeout=10) as response:
                checks[path] = response.status
        request = urllib.request.Request(
            "http://127.0.0.1:8765/api/jobs",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=10)
            checks["/api/jobs"] = "unexpected_success"
        except urllib.error.HTTPError as exc:
            checks["/api/jobs"] = exc.code
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        err_stream.close()
    write_json(root / "GPU-09" / "local_lab_check.json", checks)
    return checks


def _publish(folder: Path, schema: str, rows: list) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "index.json"
    path.write_text(json.dumps(rows, sort_keys=True) + "\n", encoding="utf-8")
    write_json(folder / "manifest.json", {"schema": schema, "files": {"index.json": file_sha256(path)}})


def run_section_52(root: Path) -> None:
    from accel import prepare_estimators
    from common import ensure_gpu_idle
    from fill import collect, write_work_plan

    from archive_release import archive_release

    log(root, "section 5.2 archive")
    archive_release(root)
    sources = prepare_estimators(root)
    first = gpu_snapshot()
    if not (gpu_is_idle(first) and (root / "GPU-00" / "gpu_sample_latest.json").is_file()):
        ensure_gpu_idle(root, first)
    else:
        latest = _load(root / "GPU-00" / "gpu_sample_latest.json")
        if not gpu_is_idle(gpu_snapshot()) or time.time() - float(latest.get("time", 0)) > 300:
            ensure_gpu_idle(root, gpu_snapshot())
    log(root, "section 5.2 SAR")
    run_sar_v2(root, sources)
    write_work_plan(collect(root))
    log(root, "section 5.2 RSS")
    run_rss_v2(root, sources)
    write_work_plan(collect(root))
    _localize(root)
    _identities(root)
    _cards(root)
    _generation(root)
    _lab(root)
    write_work_plan(collect(root))
    log(root, "section 5.2 filled")
