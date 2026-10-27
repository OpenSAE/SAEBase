"""Compat re-export. Prefer ``tasks.generate.backend``."""

from tasks.generate.vllm_backend.generate import (
    generate_vllm,
    packed_row,
    sampling_params_kwargs,
)

__all__ = ["generate_vllm", "packed_row", "sampling_params_kwargs"]
