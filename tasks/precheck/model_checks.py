"""Genuine Model-only forward Precheck, independent of any SAE checkpoint."""

from __future__ import annotations

import hashlib
from typing import Any

from tasks.precheck.report import error_result


def check_real_model(
    *,
    model_key: str,
    model_name: str,
    prompt: str,
    device: str,
) -> dict[str, Any]:
    """Load actual model weights, tokenize real input and observe finite logits."""
    try:
        import torch
        from models.hooked_lm_common import load_hooked_model

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Model Precheck requires real nonempty prompt")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("BLOCKED_RESOURCE: CUDA unavailable")
        model = load_hooked_model(model_name, device=device)
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
            raise ValueError("Unexpected native Model logit shape")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Model native logits contain NaN/Inf")
        return {
            "status": "PASS",
            "scope": "real forward with local/cache-resolved weights; no quality acceptance",
            "model_key": model_key,
            "model_name": model_name,
            "device": device,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tokens": int(tokens.numel()),
            "logit_shape": list(logits.shape),
            "scientific_acceptance": "NOT_COVERED",
        }
    except Exception as exc:
        return error_result(
            exc,
            stage=f"model:{model_key}",
            attempted=["load real Model", "tokenize", "native forward", "verify logits"],
        )
