"""Tiny deterministic saebase.examples. Every output here is synthetic test data."""

from __future__ import annotations

from types import SimpleNamespace

import torch


class ResearchToyModel(torch.nn.Module):
    """TransformerLens-style, two-site CPU-only toy model with interactions."""

    def __init__(self, *, d_model: int = 4, layer: int = 0, hook_kind: str = "hook_resid_post"):
        super().__init__()
        self.cfg = SimpleNamespace(n_layers=max(2, layer + 1), device="cpu")
        self.site_name = f"blocks.{layer}.{hook_kind}"
        self.weights = torch.nn.Parameter(torch.tensor([.7, -.3, .1, .2]))
        self.d_model = d_model

    def to_tokens(self, prompt: str) -> torch.Tensor:
        return torch.tensor([[int(x) for x in prompt.split()]], dtype=torch.long)

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.float().unsqueeze(-1)
        return torch.cat([x / 5.0, x / 7.0 + 1., torch.sin(x), torch.cos(x)], dim=-1)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # Nonlinear joint effect of the first two hidden coordinates.
        x = hidden[..., 0] * hidden[..., 1]
        y = torch.cumsum(hidden[..., 0] + .2 * hidden[..., 2], dim=1)
        return torch.stack([x + y, y - .4 * x, x - y], dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._logits(self._hidden(tokens))

    def run_with_cache(self, tokens: torch.Tensor, names_filter=None):
        return self(tokens), {self.site_name: self._hidden(tokens)} if names_filter == self.site_name else {}

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks):
        h = self._hidden(tokens)
        for name, fn in fwd_hooks:
            if name != self.site_name:
                raise ValueError(f"Unknown hook {name!r}")
            h = fn(h, None)
        return self._logits(h)


class ResearchToySAE(torch.nn.Module):
    """Linear decoder with a fixed bias and explicit non-reconstructed residual."""

    def __init__(self, *, d_in: int = 4):
        super().__init__()
        self.cfg = SimpleNamespace(d_in=d_in, d_sae=d_in + 3)
        self.bias = torch.nn.Parameter(torch.tensor([.05, -.02, .01, .03]))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, .3 * x[..., :2], torch.ones_like(x[..., :1])], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., :self.cfg.d_in] * .8 + self.bias
