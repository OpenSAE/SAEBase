"""Compatibility exports for generation; sampling is explicit in standalone SAEBase.

Legacy SAEReasoning protocol sampling constants are in its research workspace,
so callers must supply the settings when using standalone SAEBase.
"""
from __future__ import annotations

from typing import Any, Callable

from saebase.generate import generate_vllm as _generate_vllm
from saebase.generate import packed_row, sampling_params_kwargs as _sampling_params_kwargs

CompleteFn = Callable[[int, dict[str, Any]], None]
__all__ = ["generate_vllm", "packed_row", "sampling_params_kwargs"]


def sampling_params_kwargs(
    *, max_new_tokens: int, temperature: float, top_p: float
) -> dict[str, Any]:
    return _sampling_params_kwargs(
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )


def generate_vllm(
    prompts: list[str], *, max_new_tokens: int, temperature: float,
    top_p: float, on_complete: CompleteFn | None = None,
    model: str | None = None, llm: Any | None = None
) -> list[dict[str, Any]]:
    return _generate_vllm(
        prompts, max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, on_complete=on_complete, model=model, llm=llm
    )
