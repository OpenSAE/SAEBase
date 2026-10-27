"""Compat re-export. Prefer ``tasks.LLM.generate.backend``."""

from tasks.LLM.generate.vllm_backend.generate import (
    generate_vllm,
    packed_row,
    sampling_params_kwargs,
)

__all__ = ["generate_vllm", "packed_row", "sampling_params_kwargs"]
