"""Actual source-bound circuit feature edits via a native Scope 2 transcoder.

The original output is changed by decode(z_new) - decode(z), preserving
everything outside the chosen transcoder feature. This is not a proof that
a traced circuit is unique. Requires a real PyTorch forward and exact hooks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from saebase.api.scope2_native import JumpReLURepresentation, ScopeIdentity
from saebase.store import sha256_state_dict
from .graph import GraphRecord


def _module_name(hook: str) -> str:
    if not hook.endswith(".output"):
        raise ValueError("Only original output-module hooks are supported")
    return hook[: -len(".output")]


def _logits(output: Any, position: int) -> torch.Tensor:
    if hasattr(output, "logits"):
        output = output.logits
    if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] != 1:
        raise TypeError("Circuit intervention requires one [1,T,V] logit tensor")
    if not 0 <= position < output.shape[1] or not bool(torch.isfinite(output).all()):
        raise ValueError("Invalid readout token/logits")
    return output[0, position].detach().clone()


@torch.no_grad()
def execute_transcoder_node(
    model: nn.Module,
    inputs: torch.Tensor,
    graph: GraphRecord,
    node_id: str,
    transcoder: JumpReLURepresentation,
    identity: ScopeIdentity,
    *,
    new_code: float = 0.0,
    readout_position: int | None = None,
) -> dict[str, Any]:
    """Run baseline and one true module-hook feature intervention.

    This local reference supports single-layer transcoders only. A partial
    crosscoder or CLT needs simultaneous multi-source capture/target handling
    and is rejected rather than pretending its weights are an ordinary SAE.
    """
    graph.validate()
    if identity.kind != "transcoder" or transcoder.multi_layer:
        raise ValueError("This native hook runner needs a single-layer transcoder")
    if graph.transcoder_sha256 != identity.source_sha256:
        raise ValueError("Circuit and loaded transcoder checkpoint disagree")
    if sha256_state_dict(model) != graph.model_sha256:
        raise ValueError("Circuit and actual model state are different")
    if type(new_code) not in (int, float) or not torch.isfinite(torch.tensor(new_code)):
        raise ValueError("New transcoder code must be a finite scalar")
    if inputs.ndim < 2 or inputs.shape[0] != 1 or inputs.shape[1] < 1:
        raise ValueError("A single real prompt input batch is required")
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if node is None or node.kind != "feature" or node.feature_id is None:
        raise ValueError("Only actual traced feature nodes may be intervened on")
    if not 0 <= node.feature_id < transcoder.num_features:
        raise IndexError("Native transcoder feature index out of range")
    position = node.token_position
    if position is None or position >= inputs.shape[1]:
        raise ValueError("Native graph token position does not match prompt")
    source_path = _module_name(identity.input_hook)
    target_path = _module_name(identity.output_hook)
    if source_path == target_path:
        raise ValueError("Transcoder requires separate ordered input and output hooks")
    modules = dict(model.named_modules())
    if source_path not in modules or target_path not in modules:
        raise ValueError("Original official hook name is absent from actual model")
    model.eval()
    baseline = _logits(model(inputs), position if readout_position is None else readout_position)
    captured: dict[str, torch.Tensor] = {}
    fired = {"input": 0, "output": 0}
    measured_delta: torch.Tensor | None = None

    def source_hook(_module: nn.Module, _args: tuple[Any, ...], result: Any) -> None:
        if not isinstance(result, torch.Tensor) or result.ndim != 3:
            raise TypeError("Native input hook must return [B,T,D]")
        if result.shape[0] != 1 or result.shape[2] != transcoder.input_dim:
            raise ValueError("Loaded transcoder input dimension mismatch")
        fired["input"] += 1
        captured["input"] = result[:, position, :].detach().clone()

    def target_hook(_module: nn.Module, _args: tuple[Any, ...], result: Any) -> torch.Tensor:
        nonlocal measured_delta
        if "input" not in captured or fired["input"] != 1:
            raise RuntimeError("Input hook must fire before transcoder output hook")
        if not isinstance(result, torch.Tensor) or result.ndim != 3:
            raise TypeError("Transcoder target must return [B,T,D]")
        if result.shape[0] != 1 or result.shape[2] != transcoder.b_dec.shape[-1]:
            raise ValueError("Loaded transcoder output dimension mismatch")
        fired["output"] += 1
        x = captured["input"].to(device=transcoder.w_enc.device, dtype=transcoder.w_enc.dtype)
        z = transcoder.encode(x)
        z_new = z.detach().clone()
        # z has [batch, features]. Index the FEATURE axis, not batch 0:
        # z_new[0] would incorrectly erase every feature in one batch row.
        z_new[..., node.feature_id] = new_code
        # The affine/skip term cancels in the edit delta by construction.
        measured_delta = transcoder.decode(z_new) - transcoder.decode(z)
        changed = result.clone()
        changed[:, position, :] = changed[:, position, :] + measured_delta.to(
            device=changed.device, dtype=changed.dtype
        )
        return changed

    handles = [
        modules[source_path].register_forward_hook(source_hook),
        modules[target_path].register_forward_hook(target_hook),
    ]
    try:
        edited = _logits(model(inputs), position if readout_position is None else readout_position)
    finally:
        for handle in handles:
            handle.remove()
    if fired != {"input": 1, "output": 1} or measured_delta is None:
        raise RuntimeError("Each native transcoder hook must fire exactly once")
    return {
        "schema": "opensae.wp12.p6.transcoder_intervention.v1",
        "executed": True,
        "node_id": node_id,
        "feature_id": node.feature_id,
        "token_position": position,
        "graph_source_sha256": graph.source_sha256,
        "model_sha256": graph.model_sha256,
        "transcoder_sha256": graph.transcoder_sha256,
        "operation": "residual_preserving_code_replacement",
        "new_code": float(new_code),
        "activation_delta_l2": float(measured_delta.float().norm().item()),
        "logit_delta_l2": float((edited.float() - baseline.float()).norm().item()),
        "baseline_logits": baseline.detach().cpu(),
        "edited_logits": edited.detach().cpu(),
        "fired": fired,
    }
