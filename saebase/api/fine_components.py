"""Actual TransformerLens attention-head and MLP-neuron residual contributions.

Unlike block-level residual differences, these are independently computed
projection vectors for each attention head and post-nonlinearity MLP neuron.
Biases are returned separately. Unsupported architectures fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saebase.api.model import ModelAdapter, UnsupportedCapability


@dataclass(frozen=True)
class ComponentContributions:
    layer: int
    token_position: int
    component: str
    vectors: torch.Tensor
    bias: torch.Tensor
    measured_output: torch.Tensor
    reconstruction_error_max: float

    @property
    def n_components(self) -> int:
        return int(self.vectors.shape[0])

    def projected_sum(self) -> torch.Tensor:
        return self.vectors.sum(0) + self.bias


@torch.no_grad()
def capture_projected_components(
    model: ModelAdapter,
    *,
    tokens: torch.Tensor,
    layer: int,
    position: int = -1,
    component: str = "attention_heads",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> ComponentContributions:
    """Return exact per-head/MLP-neuron vectors contributing to model residual.

    Only original TransformerLens module output contractions with exposed
    parameters are supported. For gated MLP architectures, hook_post is the
    post-gating activation already passed into W_out. A head includes its
    output projection but not residual or output-normalization layers.
    """
    if model.backend != "transformer_lens":
        raise UnsupportedCapability("Component projection requires actual TransformerLens hooks")
    if type(layer) is not int or layer < 0 or component not in {"attention_heads", "mlp_neurons"}:
        raise ValueError("Supported layer and component type required")
    blocks = getattr(model.model, "blocks", None)
    if blocks is None or layer >= len(blocks):
        raise UnsupportedCapability("Selected native TransformerLens block missing")
    if component == "attention_heads":
        site = f"blocks.{layer}.attn.hook_z"
        measured_site = f"blocks.{layer}.hook_attn_out"
        module = blocks[layer].attn
        w_name, b_name = "W_O", "b_O"
    else:
        site = f"blocks.{layer}.mlp.hook_post"
        measured_site = f"blocks.{layer}.hook_mlp_out"
        module = blocks[layer].mlp
        w_name, b_name = "W_out", "b_out"
    W = getattr(module, w_name, None)
    bias = getattr(module, b_name, None)
    if not isinstance(W, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise UnsupportedCapability("Native component projection/bias tensors unavailable")
    ids = model.tokens(tokens=tokens)
    if ids.shape[0] != 1 or ids.ndim != 2:
        raise ValueError("One prompt at a time")
    length = int(ids.shape[1])
    if not -length <= position < length:
        raise IndexError("Token position outside actual sequence")
    pos = position % length
    _, cache = model.model.run_with_cache(
        ids, names_filter=lambda name: name in {site, measured_site}
    )
    if not isinstance(cache, dict) or site not in cache or measured_site not in cache:
        raise UnsupportedCapability("Requested exact native hooks did not execute")
    activation = cache[site]
    measured = cache[measured_site]
    if (
        not isinstance(activation, torch.Tensor)
        or not isinstance(measured, torch.Tensor)
        or activation.shape[0] != 1
        or measured.ndim != 3
        or measured.shape[0] != 1
        or measured.shape[1] != length
    ):
        raise RuntimeError("Component activation and output tensor shapes are invalid")
    native = activation[0, pos].float()
    weights = W.float()
    if component == "attention_heads":
        if native.ndim != 2 or weights.ndim != 3 or native.shape != weights.shape[:2]:
            raise UnsupportedCapability("Attention head projection shapes are not aligned")
        contributions = torch.einsum("hd,hdm->hm", native, weights)
    else:
        if native.ndim != 1 or weights.ndim != 2 or native.shape[0] != weights.shape[0]:
            raise UnsupportedCapability("MLP neuron projection shapes are not aligned")
        contributions = native[:, None] * weights
    if contributions.ndim != 2 or not 1 <= contributions.shape[0] <= 300000:
        raise RuntimeError("Unbounded or malformed native component decomposition")
    out = measured[0, pos].float()
    projected_bias = bias.float()
    if (
        out.shape != projected_bias.shape
        or out.shape != contributions.shape[1:]
        or not bool(torch.isfinite(contributions).all())
        or not bool(torch.isfinite(out).all())
        or not bool(torch.isfinite(projected_bias).all())
    ):
        raise RuntimeError("Projected native components are not finite and aligned")
    diff = (contributions.sum(0) + projected_bias - out).abs()
    if not torch.allclose(
        contributions.sum(0) + projected_bias,
        out,
        atol=atol,
        rtol=rtol,
    ):
        raise RuntimeError("Native component projections do not reconstitute the measured output")
    return ComponentContributions(
        layer=layer,
        token_position=pos,
        component=component,
        vectors=contributions.detach().cpu(),
        bias=projected_bias.detach().cpu(),
        measured_output=out.detach().cpu(),
        reconstruction_error_max=float(diff.max().item()),
    )
