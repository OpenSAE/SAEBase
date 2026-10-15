"""Official Qwen-Scope residual TopK checkpoint adapter.

The public checkpoints store W_enc [features, hidden] and W_dec [hidden, features].
This adapter matches Qwen-Scope's documented encode/decode equations without
requiring SAE Lens or its TransformerLens dependency to load Qwen3.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class QwenSAEConfig:
    d_in: int
    d_sae: int
    k: int
    hook_name: str


class QwenResidualTopK(nn.Module):
    """Frozen official Qwen TopK SAE, using the published orientation and biases."""

    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor, b_enc: torch.Tensor,
                 b_dec: torch.Tensor, *, k: int, hook_name: str):
        super().__init__()
        if any(not isinstance(x, torch.Tensor) for x in (W_enc, W_dec, b_enc, b_dec)):
            raise TypeError("Qwen SAE parameters must be tensors")
        if W_enc.ndim != 2 or W_dec.ndim != 2 or b_enc.ndim != 1 or b_dec.ndim != 1:
            raise ValueError("Qwen SAE tensors have invalid ranks")
        width, hidden = W_enc.shape
        if (W_dec.shape != (hidden, width) or b_enc.shape != (width,)
                or b_dec.shape != (hidden,)):
            raise ValueError("Qwen SAE encoder/decoder tensor dimensions are inconsistent")
        if type(k) is not int or not 1 <= k <= width:
            raise ValueError("Qwen TopK count must be between 1 and d_sae")
        if any(not bool(torch.isfinite(x).all()) for x in (W_enc, W_dec, b_enc, b_dec)):
            raise ValueError("Qwen SAE checkpoint contains nonfinite values")
        self.cfg = QwenSAEConfig(hidden, width, k, hook_name)
        self.W_enc = nn.Parameter(W_enc.detach().float().clone(), requires_grad=False)
        self.W_dec = nn.Parameter(W_dec.detach().float().clone(), requires_grad=False)
        self.b_enc = nn.Parameter(b_enc.detach().float().clone(), requires_grad=False)
        self.b_dec = nn.Parameter(b_dec.detach().float().clone(), requires_grad=False)
        self.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.cfg.d_in:
            raise ValueError(f"Qwen SAE expected input width {self.cfg.d_in}")
        pre = torch.nn.functional.linear(x.to(self.W_enc.dtype), self.W_enc, self.b_enc)
        values, indices = torch.topk(pre, k=self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.cfg.d_sae:
            raise ValueError(f"Qwen SAE expected feature width {self.cfg.d_sae}")
        return torch.nn.functional.linear(z.to(self.W_dec.dtype), self.W_dec, self.b_dec)


def from_official_checkpoint(
    path: str | Path, *, hook_name: str, device: str = "cpu",
    expected_hidden: int | None = None, expected_features: int | None = None,
    k: int = 50,
) -> QwenResidualTopK:
    """Load a trusted official layerN.sae.pt tensor mapping, rejecting Python pickles."""
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise ValueError("Qwen-Scope checkpoint is not a tensor mapping")
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    if not required.issubset(blob):
        raise ValueError(f"Qwen-Scope checkpoint is missing {sorted(required - set(blob))}")
    W_enc, W_dec = blob["W_enc"], blob["W_dec"]
    # Official checkpoints: [d_sae, d_in] and [d_in, d_sae], respectively.
    if not isinstance(W_enc, torch.Tensor) or W_enc.ndim != 2:
        raise ValueError("Qwen-Scope W_enc must be 2-D")
    width, hidden = W_enc.shape
    if expected_hidden is not None and hidden != expected_hidden:
        raise ValueError("Qwen SAE hidden dimension does not match model")
    if expected_features is not None and width != expected_features:
        raise ValueError("Qwen SAE feature width does not match source")
    k = int(blob.get("k", blob.get("top_k", k)))
    return QwenResidualTopK(
        W_enc, W_dec, blob["b_enc"], blob["b_dec"],
        k=k, hook_name=hook_name,
    ).to(device)
