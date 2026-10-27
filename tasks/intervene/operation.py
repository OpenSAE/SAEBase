"""Concrete intervene task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from typing import Any

from tasks.LLM.sae_runtime import _task, _pair, _prompt, _fingerprint_prompt


def _teacher_forced(cfg: Any) -> dict[str, Any]:
    from saebase.api import evaluate_edit
    import torch

    pair = _pair(cfg)
    prompt = _prompt(cfg)
    feature_id = int(_task(cfg, "feature_id", 0))
    target_value = float(_task(cfg, "value", 0.0))
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
    return {
        "status": "executed",
        "operation": "intervene",
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
    return _teacher_forced(cfg)
