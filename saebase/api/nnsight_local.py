"""Optional local NNsight execution over an existing NNsight-wrapped model.

No NDIF, remote execution, vLLM worker, implicit model download or automatic
device selection is performed here. Use an existing NNsight wrapper or call
`from_torch` explicitly. Module locations are exact user-selected paths and
the wrapper rejects ambiguous or unsafe attribute names.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

import torch


_SEGMENT = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)?(?:\[([0-9]+)\])?$")


@dataclass(frozen=True)
class TraceLocation:
    """A forward module whose output is a Tensor, or a selected tuple entry."""

    path: str
    output_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("NNsight module path cannot be empty")
        for piece in self.path.split("."):
            match = _SEGMENT.fullmatch(piece)
            if match is None or not piece or piece.startswith("_") or "__" in piece:
                raise ValueError(f"invalid NNsight module segment: {piece!r}")
        if self.output_index is not None and (
            type(self.output_index) is not int or self.output_index < 0
        ):
            raise ValueError("output_index must be a nonnegative integer")


def _resolve(model: Any, location: TraceLocation) -> Any:
    module = model
    for part in location.path.split("."):
        match = _SEGMENT.fullmatch(part)
        assert match is not None  # validated at TraceLocation construction
        name, index = match.groups()
        try:
            if name is not None:
                module = getattr(module, name)
            if index is not None:
                module = module[int(index)]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"NNsight module {location.path!r} unavailable") from exc
    return module


def _target(model: Any, location: TraceLocation) -> Any:
    node = _resolve(model, location)
    output = node.output
    if location.output_index is not None:
        output = output[location.output_index]
    return output


def _saved(tensor: Any) -> Any:
    method = getattr(tensor, "save", None)
    if not callable(method):
        raise TypeError("NNsight location must expose a traceable Tensor .save()")
    return method()


def _tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("NNsight trace must resolve to a Tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("NNsight trace produced non-finite values")
    return value.detach().clone()


class NNsightLocal:
    """Scoped local activation/gradient/edit adapter.

    `model` must be a native NNsight instance, not a raw HF nn.Module.
    Forward and backward location order is enforced by NNsight itself.
    """

    def __init__(self, model: Any) -> None:
        if not callable(getattr(model, "trace", None)):
            raise TypeError("expected an NNsight-wrapped model with .trace()")
        self.model = model

    @classmethod
    def from_torch(cls, module: torch.nn.Module) -> "NNsightLocal":
        """Wrap a preloaded PyTorch model without downloading checkpoints."""
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a loaded torch.nn.Module")
        try:
            from nnsight import NNsight
        except ImportError as exc:
            raise RuntimeError("Install the optional nnsight-local worker extra") from exc
        return cls(NNsight(module))

    def capture(self, inputs: Any, location: TraceLocation) -> torch.Tensor:
        with self.model.trace(inputs):
            result = _saved(_target(self.model, location))
        return _tensor(result)

    def capture_many(
        self,
        inputs: Sequence[Any],
        location: TraceLocation,
    ) -> tuple[torch.Tensor, ...]:
        """Use native NNsight invokers, preserving per-invocation order."""
        if not inputs:
            raise ValueError("at least one input is required")
        refs = []
        with self.model.trace() as tracer:
            for item in inputs:
                with tracer.invoke(item):
                    refs.append(_saved(_target(self.model, location)))
        if len(refs) != len(inputs):
            raise RuntimeError("NNsight invocation count changed")
        return tuple(_tensor(ref) for ref in refs)

    def edit(
        self,
        inputs: Any,
        location: TraceLocation,
        replacement: torch.Tensor,
        *,
        readout: TraceLocation,
    ) -> torch.Tensor:
        """Assign a real trace activation, then capture downstream readout.

        The user must pick an execution-order-compatible readout location.
        Replacing an SAE feature code is NOT equivalent to this native tensor
        operation; other backends use their own typed intervention contract.
        """
        if not isinstance(replacement, torch.Tensor) or not torch.isfinite(replacement).all():
            raise ValueError("replacement must be a finite Tensor")
        with self.model.trace(inputs):
            output = _target(self.model, location)
            output[:] = replacement
            changed = _saved(_target(self.model, readout))
        return _tensor(changed)

    def gradient(
        self,
        inputs: Any,
        location: TraceLocation,
        *,
        loss_location: TraceLocation,
    ) -> torch.Tensor:
        """Gradient of the sum of a downstream Tensor, via native NNsight.

        Explicitly local and autograd-capable only. The objective is a simple
        numerical smoke; research-specific objectives belong in P4/P6.
        """
        with self.model.trace(inputs):
            activation = _target(self.model, location)
            activation.requires_grad_(True)
            loss = _target(self.model, loss_location).sum()
            with loss.backward():
                grad = _saved(activation.grad)
        return _tensor(grad)
