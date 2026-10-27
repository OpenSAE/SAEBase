"""Generate-backend contract. Callers pass sampling; this module has no protocol defaults."""

from __future__ import annotations

import os
from typing import Any

CompleteFn = Any  # Callable[[int, dict[str, Any]], None] — keep import-light


def packed_row(
    *,
    prompt: str,
    token_ids: list[int],
    prompt_len: int,
    text: str,
) -> dict[str, Any]:
    """Row contract shared with ``tasks.03_logic_chain_validation.hf_runtime.generate_hf_packed``."""
    n_gen = max(0, len(token_ids) - int(prompt_len))
    return {
        "prompt": prompt,
        "token_ids": list(token_ids),
        "prompt_len": int(prompt_len),
        "token_strings": [],
        "reasoning_mask": [],
        "text": text,
        "n_gen_tokens": n_gen,
    }


def sampling_params_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """vLLM ``SamplingParams`` fields. Quantization is not a sampling field.

    ``seed`` is omitted when ``None`` so existing callers keep unseeded sampling.
    ``seed=0`` is a real seed and is forwarded.
    """
    out: dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
    }
    if seed is not None:
        out["seed"] = int(seed)
    return out


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(name: str | None = None) -> str:
    """``vllm`` or ``hf``. Default is vLLM even if this interpreter cannot import it.

    HuggingFace is **opt-in** via ``SAE_GENERATE_BACKEND=hf``. Missing ``vllm`` in
    ``/venv/main`` is not a fallback — ``generate()`` uses HTTP / autostart.
    """
    flag = (name if name is not None else os.environ.get("SAE_GENERATE_BACKEND", "")).strip().lower()
    if flag in {"hf", "huggingface"}:
        return "hf"
    return "vllm"
