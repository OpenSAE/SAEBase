"""Offline generate backend. Not a Hydra task.

Unique entry: ``generate()``. vLLM 装在当前解释器里，直接调用。
改层走 ``python -m tasks.generate.backend.nnsight_worker``。
Feature edits use the task-owned intervention implementation.
"""

from tasks.generate.backend.api import (
    packed_row,
    resolve_backend,
    sampling_params_kwargs,
    vllm_available,
)
from tasks.generate.backend.engine import generate
from tasks.generate.backend.vllm import generate_vllm

__all__ = [
    "generate",
    "generate_vllm",
    "packed_row",
    "resolve_backend",
    "sampling_params_kwargs",
    "vllm_available",
]
