"""Concrete generate task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from typing import Any

from tasks.LLM.sae_runtime import _task, _pair, _prompt, _fingerprint_prompt


def _generative(cfg: Any) -> dict[str, Any]:
    from tasks.LLM.intervene.backend.steering import GenerationSpec, compare_steered_generation
    from tasks.LLM.intervene.backend.specs import Condition, MeasurementSample
    from saebase.store import SampleRecord, TokenSelection

    pair = _pair(cfg)
    prompt = _prompt(cfg)
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
    )
    result = compare_steered_generation(
        pair.model,
        pair.sae,
        site=pair.site,
        sample=sample,
        condition=condition,
        spec=spec,
    )
    return {
        "status": "executed",
        "operation": "generate",
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "site": pair.site.hook_name,
        "prompt_sha256": _fingerprint_prompt(prompt),
        "condition": condition.to_dict(),
        "result": result,
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _generative(cfg)
