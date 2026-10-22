"""Execute a real Gemma Scope 2 CLT feature edit across downstream model layers.

This changes live PyTorch module outputs, not just a saved graph JSON or
cross-layer decoder tensor. Requires exact input/output site mappings and
the original model/transcoder identity. No arbitrary pickles are loaded.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from saebase.sae.scope2_clt import CLTIdentity, Scope2CLT
from saebase.store import sha256_state_dict

from .circuits import GraphRecord


def _module_name(template: str, layer: int) -> str:
    if template.count("{all}") != 1 or not template.endswith(".output"):
        raise ValueError("Expected original {all} module.output CLT hook template")
    return template.replace("{all}", str(layer))[: -len(".output")]


def _model_logits(value: Any, position: int) -> torch.Tensor:
    if hasattr(value, "logits"):
        value = value.logits
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] != 1
        or not 0 <= position < value.shape[1]
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("Native model must produce finite [1, tokens, vocab] logits")
    return value[0, position].detach().clone()


@torch.no_grad()
def execute_clt_feature(
    model: nn.Module,
    inputs: torch.Tensor,
    graph: GraphRecord,
    node_id: str,
    clt: Scope2CLT,
    identity: CLTIdentity,
    *,
    new_code: float = 0.0,
    readout_position: int | None = None,
) -> dict[str, Any]:
    """A selected source-layer feature edits all later native output hooks.

    The original source activation is captured before the source layer's
    MLP output. For each target layer j >= source, add only
    (z_new - z_old) * w_dec[source][feature, target].
    Earlier targets remain unchanged; biases and affine skips cancel.
    """
    if not isinstance(model, nn.Module) or not isinstance(clt, Scope2CLT):
        raise TypeError("Require actual PyTorch model and typed CLT")
    graph.validate()
    if clt.n_layers != identity.n_layers or clt.input_dim != identity.d_model:
        raise ValueError("Loaded CLT and official source identity disagree")
    if clt.features_per_layer != identity.features_per_layer:
        raise ValueError("Loaded CLT source feature width disagrees")
    if graph.transcoder_sha256 != identity.source_sha256:
        raise ValueError("Circuit and official CLT checkpoint digest disagree")
    actual_model_sha = sha256_state_dict(model)
    if actual_model_sha != graph.model_sha256:
        raise ValueError("Circuit references a different model state")
    if (
        type(new_code) not in (int, float)
        or not math.isfinite(new_code)
    ):
        raise ValueError("New feature code must be finite")
    if (
        not isinstance(inputs, torch.Tensor)
        or inputs.ndim != 3
        or inputs.shape[0] != 1
        or inputs.shape[1] < 1
    ):
        raise ValueError("Require one actual [1, T, input_dimension] input")
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if (
        node is None
        or node.kind != "feature"
        or node.layer is None
        or node.feature_id is None
        or node.token_position is None
    ):
        raise ValueError("Select a real graph feature at an exact source layer/token")
    layer, feature, token = node.layer, node.feature_id, node.token_position
    if not 0 <= layer < clt.n_layers or not 0 <= feature < clt.features_per_layer:
        raise IndexError("Native source-layer feature ID out of range")
    if not 0 <= token < inputs.shape[1]:
        raise ValueError("Graph token is absent from current input")
    out_pos = token if readout_position is None else readout_position
    if type(out_pos) is not int or not 0 <= out_pos < inputs.shape[1]:
        raise ValueError("Native target readout token position out of range")
    src_name = _module_name(identity.input_hook_template, layer)
    targets = {
        j: _module_name(identity.output_hook_template, j)
        for j in range(layer, clt.n_layers)
    }
    if src_name in targets.values():
        raise ValueError("CLT source activation and target MLP output sites must be distinct")
    modules = dict(model.named_modules())
    if src_name not in modules or any(name not in modules for name in targets.values()):
        raise ValueError("Official native CLT hook mapping is unavailable on the loaded model")
    if len(set(targets.values())) != len(targets):
        raise ValueError("Each CLT target layer requires a distinct native module")
    if clt.sources[layer].w_dec.device != next(clt.parameters(), clt.sources[layer].w_dec).device:
        raise RuntimeError("Native CLT is on inconsistent devices")

    model.eval()
    baseline = _model_logits(model(inputs), out_pos)
    state: dict[str, Any] = {"src_fired": 0, "targets": {j: 0 for j in targets}}
    handles = []

    def capture(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or output.shape[:2] != inputs.shape[:2]
            or output.shape[2] != clt.input_dim
        ):
            raise ValueError("CLT source hook activation shape is not valid")
        state["src_fired"] += 1
        if state["src_fired"] != 1:
            raise RuntimeError("CLT source hook fired multiple times")
        state["input"] = output[:, token, :].detach().clone()
        state["delta"] = clt.edit_delta(
            source_layer=layer,
            feature_id=feature,
            input_activation=state["input"],
            new_code=float(new_code),
        )

    def make_patch(target_layer: int):
        def patch(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> torch.Tensor:
            if state["src_fired"] != 1 or "delta" not in state:
                raise RuntimeError("CLT target activated before selected source")
            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 3
                or output.shape[:2] != inputs.shape[:2]
                or output.shape[2] != clt.input_dim
            ):
                raise ValueError("Native output module is not a CLT target Tensor")
            state["targets"][target_layer] += 1
            if state["targets"][target_layer] != 1:
                raise RuntimeError("CLT output target fired multiple times")
            result = output.clone()
            delta = state["delta"][:, target_layer, :].to(
                device=result.device, dtype=result.dtype
            )
            result[:, token, :] = result[:, token, :] + delta
            return result
        return patch

    try:
        handles.append(modules[src_name].register_forward_hook(capture))
        for target_layer, target_name in targets.items():
            handles.append(
                modules[target_name].register_forward_hook(make_patch(target_layer))
            )
        edited = _model_logits(model(inputs), out_pos)
    finally:
        for handle in handles:
            handle.remove()
    if state["src_fired"] != 1 or any(v != 1 for v in state["targets"].values()):
        raise RuntimeError("Not every requested native CLT module fired exactly once")
    delta_logits = edited.float() - baseline.float()
    if not bool(torch.isfinite(delta_logits).all()):
        raise RuntimeError("Native cross-layer edits produced nonfinite logits")
    return {
        "schema": "opensae.wp12.p6.clt_feature_edit.v1",
        "executed": True,
        "source_layer": layer,
        "feature_id": feature,
        "token_position": token,
        "target_layers": list(targets),
        "fired": dict(state["targets"]),
        "model_sha256": actual_model_sha,
        "clt_source_sha256": identity.source_sha256,
        "graph_sha256": graph.source_sha256,
        "edit_semantics": "decoder_feature_delta_triangular_targets",
        "activation_delta_l2": float(state["delta"].float().norm().item()),
        "logit_delta_l2": float(delta_logits.norm().item()),
        "baseline_logits": baseline.detach().cpu(),
        "edited_logits": edited.detach().cpu(),
    }
