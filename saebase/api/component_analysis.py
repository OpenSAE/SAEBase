"""Block-level residual contributions from actual supported hook locations.

These are residual-state differences, not direct logit attribution or
independent causal effects. The calculation explicitly requires all three
hook points to be available in the selected execution backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from saebase.api.model import ActivationSite, ModelAdapter


@dataclass(frozen=True)
class BlockComponents:
    layer: int
    position: int
    residual_before: torch.Tensor
    attention_delta: torch.Tensor
    mlp_delta: torch.Tensor
    residual_after: torch.Tensor

    def check(self, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
        if not torch.allclose(
            self.residual_before + self.attention_delta + self.mlp_delta,
            self.residual_after,
            atol=atol, rtol=rtol,
        ):
            raise RuntimeError("block residual components do not reconstitute the output")


@torch.no_grad()
def capture_block_components(
    model: ModelAdapter, *, tokens: torch.Tensor, layer: int, position: int = -1,
) -> BlockComponents:
    """Capture before/mid/after at a single token and return additive deltas.

    Depending on architecture, the difference called `attention_delta` may
    include work between the two residual hooks. It is not assumed equal to a
    particular head or MLP module output. On unsupported hooks, raise rather
    than substituting zeros or reporting an invented decomposition.
    """
    before = model.capture(
        tokens=tokens, site=ActivationSite(layer, f"blocks.{layer}.hook_resid_pre", position)
    ).float()
    middle = model.capture(
        tokens=tokens, site=ActivationSite(layer, f"blocks.{layer}.hook_resid_mid", position)
    ).float()
    after = model.capture(
        tokens=tokens, site=ActivationSite(layer, f"blocks.{layer}.hook_resid_post", position)
    ).float()
    if before.shape != middle.shape or middle.shape != after.shape:
        raise ValueError("block hooks disagree on residual shape")
    result = BlockComponents(
        layer=layer, position=position, residual_before=before,
        attention_delta=middle - before, mlp_delta=after - middle,
        residual_after=after,
    )
    result.check()
    return result
