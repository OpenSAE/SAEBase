"""Capture-owned Top N selection for per-feature intervene/generate comparisons.

Uses the native capture operation and its normalization; saved capture rankings
can also be reused with exact prompt, checkpoint and site identity checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from omegaconf import OmegaConf


def validate_top_selection(cfg):
    if cfg.task.get("feature_selection", "configured") != "capture_top":
        return
    for key in ("top_feature_num", "whiten_chunk_size"):
        value = cfg.task.get(key, 10 if key == "top_feature_num" else 256)
        if type(value) is not int or value < 1:
            raise ValueError(f"task.{key} must be a positive integer")
    from tasks.sae_runtime import validate_fields

    validate_fields(
        cfg, booleans=("resume", "inline_results", "continue_on_error", "reuse_prompt_observations")
    )
    if cfg.task.get("feature_ranking", "non_whiten") not in {"non_whiten", "whiten"}:
        raise ValueError("task.feature_ranking must be non_whiten or whiten")
    if cfg.task.get("protocol") is not None or cfg.task.get("conditions"):
        raise ValueError(
            "capture_top uses independent single-feature edits; omit protocol/conditions"
        )
    if cfg.dataset.get("samples") is not None:
        raise ValueError(
            "capture_top requires Dataset prompts, not dataset.samples token protocols"
        )
    if cfg.task.get("capture_source") and cfg.task.get("execute", False):
        from tasks.sae_runtime import input_path

        source = input_path(cfg.task.capture_source)
        if not (source / "reports/capture_samples.jsonl").is_file():
            raise FileNotFoundError("task.capture_source must name a saved capture run")


def capture_top_cohort(cfg, pair):
    """Yield each prompt with the exact capture-ranked features, before any edits."""
    from importlib.resources import files

    from saebase.api import capture_features
    from saebase.datasets.selection import select_dataset
    from tasks.capture.operation import _capture
    from tasks.sae_runtime import _fingerprint_prompt, pair_state_fingerprints

    count = int(cfg.task.get("top_feature_num", 10))
    filter_context = None
    mode = str(cfg.task.get("feature_ranking", "non_whiten"))
    query = str(cfg.task.get("prompt", "") or "").strip()
    if query and (mode == "non_whiten" or cfg.task.get("capture_source")):
        reference = []
    else:
        reference = list(
            select_dataset(
                cfg.dataset,
                model_key=pair.identity.model_key,
                model=pair.model.model,
                loaded_model_name=pair.identity.loaded_model_name,
                seed=int(cfg.seed),
            )
        )
    rows = (
        [{"sample_id": "explicit:" + _fingerprint_prompt(query), "prompt": query}]
        if query
        else reference
    )
    if not rows:
        raise ValueError("capture_top requires a nonempty prompt cohort")
    capture_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    capture_cfg.task = OmegaConf.load(files("conf") / "task/capture.yaml").task
    for key in (
        "sae_key",
        "layer",
        "device",
        "prompt",
        "top_feature_num",
        "whiten_chunk_size",
        "neuronpedia",
        "filter",
        "capture_count",
        "sar",
    ):
        if key in cfg.task:
            capture_cfg.task[key] = cfg.task[key]
    capture_cfg.task.mode = mode
    # Historical feature-task configs did not opt into a candidate filter.
    # Forward new explicit settings, preserving those existing configurations.
    if "filter" not in cfg.task:
        del capture_cfg.task.filter
        del capture_cfg.task.capture_count
    normalization = None
    snapshots = {}
    saved = None
    source_info = None
    if cfg.task.get("capture_source"):
        from tasks.capture.annotations import read_samples
        from tasks.sae_runtime import input_path

        source = input_path(cfg.task.capture_source)
        ledger = source / "reports/capture_samples.jsonl"
        source_info = {
            "path": str(source),
            "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        }
        saved = {}
        for entry in read_samples(source):
            result = entry.get("result", {})
            if "top_features" in result:
                key = result["prompt_sha256"]
                if key in saved and saved[key]["top_features"] != result["top_features"]:
                    raise ValueError("Saved capture contains conflicting rankings for one prompt")
                saved[key] = result
    else:
        from tasks.capture.selection import (
            prepare_filter,
            needs_normalization,
            reference_rows,
            fit_reference,
        )

        filter_context = prepare_filter(capture_cfg)
        if needs_normalization(capture_cfg, filter_context):
            reference = reference_rows(capture_cfg, pair, reference, filter_context)
            normalization, snapshots = fit_reference(
                capture_cfg, pair, reference, filter_context, Path.cwd() / "metadata"
            )
    model_sha, sae_sha = pair_state_fingerprints(pair)
    for row in rows:
        prompt = str(row.get("prompt", ""))
        if not prompt.strip():
            raise ValueError(f"Empty capture_top prompt: {row['sample_id']}")
        if saved is not None:
            result = saved.get(_fingerprint_prompt(prompt))
            if result is None:
                raise ValueError(f"No matching prompt in task.capture_source: {row['sample_id']}")
            actual = (
                result.get("model_key"),
                result.get("sae_key"),
                result.get("layer"),
                result.get("site"),
            )
            expected = (
                pair.identity.model_key,
                pair.identity.sae_key,
                pair.site.layer,
                pair.site.hook_name,
            )
            identity = result.get("input_identity") or {}
            ranking = "absolute_zscore" if mode == "whiten" else "absolute_native_activation"
            if actual != expected or result.get("ranking", "absolute_native_activation") != ranking:
                raise ValueError("Saved capture model/SAE/site/ranking differs from this task")
            if identity.get("model_sha256") != model_sha or identity.get("sae_sha256") != sae_sha:
                raise ValueError("Saved capture checkpoint fingerprints differ or are missing")
        else:
            snapshot = snapshots.get((row["sample_id"], prompt))
            if snapshot is None:
                snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
            result = _capture(
                capture_cfg,
                pair=pair,
                prompt=prompt,
                snapshot=snapshot,
                normalization=normalization,
                sar_context=filter_context,
            )
        features = result["top_features"][:count]
        if len(features) != count or len({f["feature_id"] for f in features}) != count:
            raise ValueError(f"capture_top requires {count} distinct features; got {len(features)}")
        yield {
            "sample_id": row["sample_id"],
            "prompt": prompt,
            "prompt_sha256": _fingerprint_prompt(prompt),
            "top_features": features,
            "ranking": result.get("ranking", "absolute_native_activation"),
            "normalization": result.get("normalization"),
            "input_identity": result.get("input_identity"),
            "capture_source": source_info,
        }
