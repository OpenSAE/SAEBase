"""Lazy vLLM offline generate. Do not import this module's vLLM symbols at saebase import time."""

from __future__ import annotations

from typing import Any, Callable

from saebase.generate.api import packed_row, sampling_params_kwargs

CompleteFn = Callable[[int, dict[str, Any]], None]


def generate_vllm(
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    on_complete: CompleteFn | None = None,
    model: str | None = None,
    llm: Any | None = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    dtype: str = "bfloat16",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate ``prompts`` with vLLM offline API.

    ``llm`` may be a pre-built ``vllm.LLM`` so callers reuse one engine.
    Chat wrapping stays in the caller. ``model`` is required when ``llm`` is None.
    """
    if not prompts:
        return []
    from saebase.generate.settings import apply_process_env

    apply_process_env()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this interpreter. "
            "Install them using requirements-generation.txt in this interpreter."
        ) from exc

    if llm is None and not model:
        raise ValueError("generate_vllm requires model= or llm=")

    engine = llm if llm is not None else LLM(
        model=str(model),
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        **sampling_params_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    )
    outputs = engine.generate(list(prompts), params)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
        )
    tok = getattr(engine, "get_tokenizer", lambda: None)()
    rows: list[dict[str, Any]] = []
    for i, (prompt, out) in enumerate(zip(prompts, outputs)):
        seq = out.outputs[0]
        gen_ids = [int(t) for t in (seq.token_ids or [])]
        if tok is not None:
            prompt_ids = [int(t) for t in tok.encode(prompt, add_special_tokens=False)]
        else:
            prompt_ids = []
        token_ids = prompt_ids + gen_ids
        text = seq.text if seq.text else (
            tok.decode(gen_ids, skip_special_tokens=False) if tok is not None else ""
        )
        row = packed_row(
            prompt=prompt,
            token_ids=token_ids,
            prompt_len=len(prompt_ids),
            text=text,
        )
        if on_complete is not None:
            on_complete(i, row)
        rows.append(row)
    return rows
