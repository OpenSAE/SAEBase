"""Concrete capture task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from typing import Any

from tasks.LLM.sae_runtime import _pair, _prompt, _fingerprint_prompt


def _capture(cfg: Any) -> dict[str, Any]:
    from saebase.api import capture_features
    import torch

    pair = _pair(cfg)
    prompt = _prompt(cfg)
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
    codes = snapshot.codes.detach().float().cpu()
    top = torch.topk(codes.abs(), k=min(10, codes.numel()))
    return {
        "status": "executed",
        "operation": "capture",
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "site": pair.site.hook_name,
        "prompt_sha256": _fingerprint_prompt(prompt),
        "feature_count": int(codes.numel()),
        "active_count": int(torch.count_nonzero(codes).item()),
        "top_features": [
            {"feature_id": int(i), "activation": float(codes[int(i)].item())}
            for i in top.indices.tolist()
        ],
        "reconstruction_l2": float(
            torch.linalg.vector_norm(snapshot.reconstruction.float()).item()
        ),
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _capture(cfg)
