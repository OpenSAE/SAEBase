"""Compatibility exports backed by the canonical SAEBase generation API."""
from saebase.generate.api import packed_row, resolve_backend, sampling_params_kwargs, vllm_available

__all__ = ["packed_row", "resolve_backend", "sampling_params_kwargs", "vllm_available"]
