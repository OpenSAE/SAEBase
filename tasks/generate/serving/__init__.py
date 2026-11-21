"""Offline generate backend. Not a Hydra task.

Unique entry: ``generate()``. vLLM Put it in the current interpreter and call it directly..
Let’s change the floor. ``python -m tasks.generate.serving.nnsight_worker``.
Feature edits use the task-owned intervention implementation.
"""

from tasks.generate.serving.service import (
    packed_row,
    resolve_backend,
    sampling_params_kwargs,
    vllm_available,
)
from tasks.generate.serving.engine import generate
from tasks.generate.serving.serving_engine import generate_vllm

__all__ = [
    "generate",
    "generate_vllm",
    "packed_row",
    "resolve_backend",
    "sampling_params_kwargs",
    "vllm_available",
]
