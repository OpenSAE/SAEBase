"""SAEBase dense-code adapter for EleutherAI's native sparsify checkpoints.

Delegates TopK activation selection to the original sparsify SparseCoder.
Only converts its native (values, indices) representation into a dense code
vector for the existing SAEBase API; no trained weights are transformed.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


class SparsifyDenseAdapter(nn.Module):
    """Expose cfg.d_in/d_sae and dense encode/decode to SAEBase consumers."""

    def __init__(self, native: nn.Module, *, hook_name: str):
        super().__init__()
        if not hasattr(native, "encoder") or not hasattr(native, "W_dec"):
            raise TypeError("expected original EleutherAI sparsify SparseCoder")
        if getattr(native.cfg, "transcode", False) or getattr(native.cfg, "skip_connection", False):
            raise ValueError("Sparsify transcoders/skip connections need their own adapter")
        if str(getattr(native.cfg, "activation", "relu")) not in ("topk", "relu"):
            raise ValueError("only native TopK SparseCoder is supported here")
        if getattr(native, "W_dec", None) is None:
            raise ValueError("Sparsify checkpoint lacks a decoder")
        self.native = native
        self.cfg = SimpleNamespace(
            d_in=int(native.d_in),
            d_sae=int(native.num_latents),
            hook_name=hook_name,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.cfg.d_in:
            raise ValueError("Sparsify input must be [N, d_in]")
        values, indices, _ = self.native.encode(x)
        if values.shape != indices.shape or values.ndim != 2:
            raise ValueError("invalid original Sparsify sparse code")
        if not torch.isfinite(values).all():
            raise ValueError("original Sparsify produced nonfinite activations")
        dense = torch.zeros(
            (len(x), self.cfg.d_sae), dtype=values.dtype, device=values.device
        )
        dense.scatter_(1, indices.long(), values)
        return dense

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.cfg.d_sae:
            raise ValueError("Sparsify decoder input must be [N, d_sae]")
        # The original decoder is sum_j z_j W_dec[j] + b_dec. Using the dense
        # equivalent preserves arbitrary feature edits and sign of interventions.
        return z.to(dtype=self.native.W_dec.dtype) @ self.native.W_dec + self.native.b_dec


def load_sparsify_topk(
    *,
    repo_id: str,
    hookpoint: str,
    hook_name: str,
    device: str,
) -> tuple[SparsifyDenseAdapter, dict[str, Any]]:
    try:
        from sparsify import Sae
    except ImportError as exc:
        raise ImportError(
            "Install the original EleutherAI sparsify library: "
            "python -m pip install 'eai-sparsify==1.3.3'"
        ) from exc
    native = Sae.load_from_hub(repo_id, hookpoint=hookpoint, device=device)
    wrapped = SparsifyDenseAdapter(native, hook_name=hook_name).eval()
    for param in wrapped.parameters():
        param.requires_grad_(False)
    cfg = {
        "d_in": wrapped.cfg.d_in,
        "d_sae": wrapped.cfg.d_sae,
        "k": int(native.cfg.k),
        "source": "original_eai_sparsify",
        "repo_id": repo_id,
        "hookpoint": hookpoint,
    }
    return wrapped, cfg
