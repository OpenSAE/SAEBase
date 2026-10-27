"""Prompt → hook activation → SAE encode/decode analysis."""

from __future__ import annotations

from typing import Any

import torch

from saebase.sae.contracts import (
    assert_activation_sae_shapes,
    assert_finite_tensors,
    assert_prompt_record,
    latent_l0,
    reconstruction_stats,
)


@torch.no_grad()
def analyze_prompt_activation(
    *,
    model: Any,
    sae: Any,
    hook_name: str,
    ex: dict[str, Any],
    prompt_idx: int,
) -> dict[str, Any]:
    """Run one prompt through model hook + SAE; return health metrics."""
    ctx = f"prompt_idx={prompt_idx}"
    assert_prompt_record(ex, model=model, ctx=ctx)

    prompt = str(ex["prompt"])
    tok = ex.get("tokenization") if isinstance(ex.get("tokenization"), dict) else {}
    if isinstance(tok.get("input_ids"), list) and tok["input_ids"]:
        tokens = torch.tensor([tok["input_ids"]], device=next(model.parameters()).device)
    else:
        tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    if hook_name not in cache:
        raise RuntimeError(f"{ctx}: hook {hook_name!r} missing from cache")

    h = cache[hook_name][0, -1].float()
    z = sae.encode(h.unsqueeze(0))[0].float()
    h_hat = sae.decode(z.unsqueeze(0))[0].float()

    assert_activation_sae_shapes(h, z, sae, ctx=ctx)
    assert_finite_tensors(h, z, h_hat, ctx=ctx)
    l0 = latent_l0(z)
    if l0 <= 0:
        raise RuntimeError(f"{ctx}: degenerate SAE latent L0={l0}")

    recon = reconstruction_stats(h, h_hat)
    return {
        "prompt_idx": prompt_idx,
        "prompt_id": ex.get("id", f"idx-{prompt_idx}"),
        "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "token_count": int(tokens.shape[1]),
        "activation_shape": list(h.shape),
        "activation_norm": float(h.norm().item()),
        "d_in": int(sae.cfg.d_in),
        "latent_shape": list(z.shape),
        "d_sae": int(sae.cfg.d_sae),
        "l0": l0,
        **recon,
        "ok": True,
    }
