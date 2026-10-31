"""Task-owned real SAE capture, independent of experiment identity."""
from __future__ import annotations
import math
from typing import Any
from saebase.integrations.neuronpedia import annotate

def _capture_one(
    pair: Any,
    tokens: Any,
    *,
    top_k: int,
    mapping: dict[str, Any],
    annotations: dict[str, Any],
    sae_sha256: str | None,
    site_id: str,
) -> dict[str, Any]:
    import torch
    from saebase.api import capture_features

    site = pair.site
    with torch.inference_mode():
        snap = capture_features(pair.model, pair.sae, tokens=tokens, site=site)
        codes = snap.codes.detach().float().cpu()
        active = torch.nonzero(codes != 0, as_tuple=False).flatten()
        top = sorted(
            (int(i) for i in active.tolist()),
            key=lambda i: (-abs(float(codes[i].item())), i),
        )[:top_k]
        h = snap.activation.detach().float().cpu()
        recon = snap.reconstruction.detach().float().cpu()
        mse = float(torch.mean((h - recon) ** 2).item())
        norm_h = float(torch.linalg.vector_norm(h).item())
        logit = pair.model.forward_logits(tokens=tokens)
        if not bool(torch.isfinite(logit).all()):
            raise RuntimeError("Model output contains non-finite logits")
    if not math.isfinite(mse) or len(codes) != pair.identity.d_sae:
        raise RuntimeError("Invalid SAE feature dimensions or reconstruction")
    if any(not math.isfinite(float(codes[i])) for i in active):
        raise RuntimeError("Non-finite active SAE feature")
    return {
        "token_count": int(tokens.shape[1]),
        "sae_d_in": pair.identity.d_in,
        "sae_d_sae": pair.identity.d_sae,
        "active_count": int(active.numel()),
        "l0": int(active.numel()),
        "reconstruction_mse": mse,
        "activation_norm_l2": norm_h,
        "baseline_argmax_id": int(torch.argmax(logit).item()),
        "baseline_logits_finite": True,
        "capture_position": -1,
        "top_features": [
            {
                **annotate(
                    i,
                    site_id=site_id,
                    sae_sha256=sae_sha256,
                    mapping=mapping,
                    annotations=annotations,
                ),
                "activation": float(codes[i].item()),
            }
            for i in top
        ],
    }


