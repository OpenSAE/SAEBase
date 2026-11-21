"""Concrete intervene task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value

from typing import Any

from saebase.runtime.sae_operations import _task, _pair, _prompt, _fingerprint_prompt


def _teacher_forced(cfg: Any, *, prompt: str | None = None) -> dict[str, Any]:
    from saebase.api import evaluate_edit
    import torch

    pair = _pair(cfg)
    prompt = _prompt(cfg) if prompt is None else prompt
    feature_id = int(_task(cfg, "feature_id", 0))
    target_value = float(_task(cfg, "value", 0.0))
    from tasks.intervene.editing.observations import current_observations

    observations = current_observations(pair.model, pair.sae)
    if observations is not None:
        outcome = observations.evaluate(
            prompt, {feature_id: target_value}, bool(_task(cfg, "verify_identity", True))
        )
    else:
        outcome = evaluate_edit(
            pair.model,
            pair.sae,
            site=pair.site,
            prompt=prompt,
            replacements={feature_id: target_value},
            verify_identity=bool(_task(cfg, "verify_identity", True)),
        )
    before = outcome.baseline_logits.detach().float()
    after = outcome.edited_logits.detach().float()
    delta = after - before
    from saebase.results import completion_summary

    return {
        "completion": completion_summary([{"status": "completed"}]),
        "status": "executed",
        "operation": "intervene",
        "input_scope": {
            "unit": "prompt",
            "requested": 1,
            "processed": 1,
            "source": "explicit_prompt" if cfg.task.get("prompt") else "first_dataset_record",
        },
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "site": pair.site.hook_name,
        "prompt_sha256": _fingerprint_prompt(prompt),
        "feature_id": feature_id,
        "replacement_value": target_value,
        "logit_delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "max_abs_logit_change": float(delta.abs().max().item()),
        "top_baseline_token_ids": [
            int(x) for x in torch.topk(before, min(5, before.numel())).indices.tolist()
        ],
        "top_edited_token_ids": [
            int(x) for x in torch.topk(after, min(5, after.numel())).indices.tolist()
        ],
    }


def execute(cfg: Any) -> dict[str, Any]:
    if task_value(cfg.task, "feature_selection", "configured") == "capture_top":
        from tasks.intervene.top_features import run_top_features

        return run_top_features(cfg, "intervene", _teacher_forced)
    if task_value(cfg.task, "protocol") is not None:
        from pathlib import Path

        from tasks.intervene.batch import configured_plan, run_batch

        return run_batch(_pair(cfg), configured_plan(cfg), Path.cwd() / "reports", resume=True)
    return _teacher_forced(cfg)


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from saebase.runtime.sae_operations import validate_fields, validate_sae_selection

    validate_fields(
        cfg,
        integers=(("feature_id", 0),),
        numbers=(("value", None),),
        booleans=("verify_identity",),
    )
    validate_sae_selection(cfg)
    from tasks.capture.top_features import validate_top_selection

    if task_value(cfg.task, "feature_selection", "configured") not in {"configured", "capture_top"}:
        raise ValueError("feature_selection must be configured or capture_top")
    validate_top_selection(cfg)
    if task_value(cfg.task, "protocol") is not None:
        from tasks.intervene.batch import configured_plan, parse_plan

        parse_plan(configured_plan(cfg))


def prepare_execution(cfg: Any) -> dict:
    from tasks.intervene.batch import configured_plan, parse_plan
    from saebase.api import capture_features
    import torch

    _, samples, _, _, _, _ = parse_plan(configured_plan(cfg))
    if not samples:
        raise ValueError("Intervention requires nonempty token samples")
    pair = _pair(cfg)
    capture_features(pair.model, pair.sae, site=pair.site, tokens=torch.tensor([samples[0].tokens]))
    return {"status": "PASS", "scope": "first token sample captured; batch protocol prepared"}
