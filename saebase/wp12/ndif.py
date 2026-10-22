"""P7 genuinely remote NNsight trace opt-in.

Never turns a local simulation or self-hosted worker into an NDIF success.
The user supplies an authorized NNsight model instance deployed on NDIF.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import torch

from saebase.api.nnsight_local import TraceLocation, _target


def remote_ndif_capture(
    model: Any, prompt: str, location: TraceLocation, *, model_id: str,
    approved: bool = False, rights_reviewed: bool = False,
    deployment_allowlist: frozenset[str] = frozenset(),
    max_output_elements: int = 2048,
) -> dict[str, Any]:
    """Execute one real NDIF remote=True trace after explicit consent.

    If the deployed model or credentials are unknown, fail before submission.
    No NDIF account is created and no external GPU costs are authorized here.
    """
    if not approved or not rights_reviewed:
        raise PermissionError("Remote transmission of user text requires explicit approval")
    if not os.environ.get("NDIF_API_KEY"):
        raise PermissionError("NDIF_API_KEY is not configured in this worker")
    if not isinstance(model_id, str) or model_id not in deployment_allowlist:
        raise PermissionError("Model must be in an explicitly authorized NDIF deployment list")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 2000:
        raise ValueError("NDIF prompt must be a bounded input")
    if not isinstance(location, TraceLocation):
        raise TypeError("Select an exact trace location")
    if not 1 <= max_output_elements <= 4096:
        raise ValueError("Bounded remote transfer required")
    if not callable(getattr(model, "trace", None)):
        raise TypeError("Use a real Remoteable NNsight model, not a raw PyTorch module")
    # No remote='local' substitution: this path is actual NDIF.
    with model.trace(prompt, remote=True, blocking=True):
        output = _target(model, location)
        result = output.detach().flatten()[:max_output_elements].cpu().save()
    if not isinstance(result, torch.Tensor) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("Remote NDIF did not return a finite saved Tensor")
    return {
        "mode": "ndif_remote",
        "model_id": model_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "site": location.path,
        "output_shape": list(result.shape),
        "output": result.detach().clone(),
        "max_returned_elements": max_output_elements,
        "status": "remote_tensor_received_not_independently_verified",
    }
