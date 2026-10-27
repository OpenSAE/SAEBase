"""Offline generate backend. Not a Hydra task.

Unique entry: ``generate()``. vLLM 装在当前解释器里，直接调用。
改层走 ``python -m tasks.LLM.generate.backend.nnsight_worker``。
HuggingFace packed decode stays in
``tasks.03_logic_chain_validation.hf_runtime``，不会被静默选中。
"""

from tasks.LLM.generate.backend.api import packed_row, resolve_backend, sampling_params_kwargs, vllm_available
from tasks.LLM.generate.backend.engine import generate
from tasks.LLM.generate.backend.vllm import generate_vllm

__all__ = [
    "generate",
    "generate_vllm",
    "packed_row",
    "resolve_backend",
    "sampling_params_kwargs",
    "vllm_available",
]
