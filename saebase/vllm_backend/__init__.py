"""Compat re-export. Prefer ``saebase.generate``."""

from saebase.vllm_backend.generate import generate_vllm, packed_row, sampling_params_kwargs

__all__ = ["generate_vllm", "packed_row", "sampling_params_kwargs"]
