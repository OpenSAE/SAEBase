"""Concrete generate task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from typing import Any

from tasks.sae_runtime import _task, _pair, _prompt, _fingerprint_prompt


def _generative(cfg: Any, *, prompt: str | None = None) -> dict[str, Any]:
    from tasks.intervene.backend.steering import GenerationSpec, compare_steered_generation
    from tasks.intervene.backend.specs import Condition, MeasurementSample
    from saebase.store import SampleRecord, TokenSelection

    pair = _pair(cfg)
    prompt = _prompt(cfg) if prompt is None else prompt
    if cfg.task.get(
        "feature_selection", "configured"
    ) == "last_active_content_token" or cfg.task.get("conditions"):
        from tasks.generate.conditions import compare_prompt_conditions
        from saebase.results import completion_summary
        from omegaconf import OmegaConf

        conditions = cfg.task.get("conditions")
        result = compare_prompt_conditions(
            pair,
            prompt,
            "prompt-" + _fingerprint_prompt(prompt)[:16],
            int(cfg.task.get("max_new_tokens", 8)),
            feature_id=None
            if cfg.task.get("feature_selection") == "last_active_content_token"
            else cfg.task.get("feature_id", 0),
            conditions=OmegaConf.to_container(conditions, resolve=True) if conditions else None,
            seed=int(cfg.seed),
            temperature=float(cfg.task.get("temperature", 0.0)),
            stop_at_context_limit=cfg.task.get("stop_at_context_limit", True),
        )
        return {
            "status": "executed",
            "operation": "generate",
            "model_key": pair.identity.model_key,
            "sae_key": pair.identity.sae_key,
            "site": pair.site.hook_name,
            "result": result,
            "input_scope": {"unit": "prompt", "requested": 1, "processed": 1},
            "completion": completion_summary(
                [{"status": "completed"} for _ in result["runs"]], unit="generation_condition"
            ),
        }
    tokens = pair.model.tokens(prompt=prompt)
    token_ids = tuple(int(token) for token in tokens[0].detach().cpu().tolist())
    if not token_ids:
        raise ValueError("No tokens produced by the selected model")
    record = SampleRecord(
        sample_id="prompt-" + _fingerprint_prompt(prompt)[:16],
        token_ids=token_ids,
        selection=TokenSelection(mode="last_valid"),
        prompt_group_id=_fingerprint_prompt(prompt),
    )
    sample = MeasurementSample(record, readout_position=-1)
    operation = str(_task(cfg, "intervention", "add"))
    condition = Condition(
        name="generation_feature_edit",
        operation=operation,
        feature_ids=(int(_task(cfg, "feature_id", 0)),),
        value=float(_task(cfg, "value", 1.0)),
    )
    spec = GenerationSpec(
        max_new_tokens=int(_task(cfg, "max_new_tokens", 8)),
        seed=int(cfg.seed),
        temperature=float(_task(cfg, "temperature", 0.0)),
        stop_at_context_limit=cfg.task.get("stop_at_context_limit", True),
    )
    result = compare_steered_generation(
        pair.model,
        pair.sae,
        site=pair.site,
        sample=sample,
        condition=condition,
        spec=spec,
    )
    from saebase.results import completion_summary

    return {
        "completion": completion_summary([{"status": "completed"}]),
        "status": "executed",
        "operation": "generate",
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
        "condition": condition.to_dict(),
        "result": result,
    }


def execute(cfg: Any) -> dict[str, Any]:
    if cfg.task.get("feature_selection", "configured") == "capture_top":
        from tasks.intervene.top_features import run_top_features

        return run_top_features(cfg, "generate", _generative)
    return _generative(cfg)


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from tasks.sae_runtime import validate_sae_selection
    from tasks.intervene.backend.specs import Condition
    from tasks.intervene.backend.steering import GenerationSpec

    Condition(
        "generation_feature_edit",
        str(cfg.task.get("intervention", "add")),
        (cfg.task.get("feature_id", 0),),
        cfg.task.get("value", 1.0),
    )
    GenerationSpec(
        max_new_tokens=cfg.task.get("max_new_tokens", 8),
        seed=cfg.seed,
        temperature=cfg.task.get("temperature", 0.0),
        stop_at_context_limit=cfg.task.get("stop_at_context_limit", True),
    )
    if cfg.task.get("feature_selection", "configured") not in {
        "configured",
        "last_active_content_token",
        "capture_top",
    }:
        raise ValueError(
            "feature_selection must be configured, last_active_content_token or capture_top"
        )
    from tasks.capture.top_features import validate_top_selection

    validate_top_selection(cfg)
    if cfg.task.get("conditions"):
        for row in cfg.task.conditions:
            Condition(
                row.name,
                row.operation,
                tuple(
                    row.get(
                        "feature_ids",
                        ()
                        if row.operation in {"identity", "reconstruction"}
                        else (cfg.task.get("feature_id", 0),),
                    )
                ),
                row.get("value", 0.0),
            )
    validate_sae_selection(cfg)
