"""Public generate() — vLLM in-process or HTTP. No silent HuggingFace fallback."""

from __future__ import annotations

from typing import Any, Callable

from saebase.generate.api import packed_row, resolve_backend, vllm_available
from saebase.generate.vllm import generate_vllm

CompleteFn = Callable[[int, dict[str, Any]], None]

_engine: Any = None
_engine_settings: tuple[str, float, int, str] | None = None
_tok: Any = None
_tok_model: str | None = None


def release() -> None:
    """丢掉本进程里缓存的 vLLM 引擎，显存交给下一次加载。"""
    global _engine, _engine_settings
    _engine = None
    _engine_settings = None


def _cached_llm(model: str, *, gpu_memory_utilization: float, max_model_len: int, dtype: str) -> Any:
    global _engine, _engine_settings
    requested = (str(model), float(gpu_memory_utilization), int(max_model_len), str(dtype))
    if _engine is not None and _engine_settings == requested:
        return _engine
    if _engine is not None:
        release()
    from saebase.generate.settings import apply_process_env

    apply_process_env()
    from vllm import LLM

    _engine = LLM(
        model=str(model),
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        enable_prefix_caching=True,
    )
    _engine_settings = requested
    return _engine


def _http_tokenizer(model: str) -> Any:
    global _tok, _tok_model
    if _tok_model == str(model):
        return _tok
    try:
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    except Exception:
        _tok = None
    _tok_model = str(model)
    return _tok


def _rows_from_http(
    prompts: list[str],
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    on_complete: CompleteFn | None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    from saebase.generate.server import post_completions

    body = {
        "model": str(model),
        "prompt": list(prompts),
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "skip_special_tokens": False,
    }
    if seed is not None:
        body["seed"] = int(seed)
    payload = post_completions(body)
    choices = sorted(payload.get("choices") or [], key=lambda c: int(c.get("index") or 0))
    if len(choices) != len(prompts):
        raise RuntimeError(
            f"vLLM HTTP returned {len(choices)} choices for {len(prompts)} prompts"
        )
    tok = _http_tokenizer(model)
    rows: list[dict[str, Any]] = []
    for i, (prompt, ch) in enumerate(zip(prompts, choices)):
        text = str(ch.get("text") or "")
        if tok is not None:
            prompt_ids = [int(t) for t in tok.encode(prompt, add_special_tokens=False)]
            gen_ids = [int(t) for t in tok.encode(text, add_special_tokens=False)]
        else:
            prompt_ids, gen_ids = [], []
        row = packed_row(
            prompt=prompt,
            token_ids=prompt_ids + gen_ids,
            prompt_len=len(prompt_ids),
            text=text,
        )
        if on_complete is not None:
            on_complete(i, row)
        rows.append(row)
    return rows


def generate(
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model: str,
    on_complete: CompleteFn | None = None,
    llm: Any | None = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    dtype: str = "bfloat16",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample ``prompts``. Default backend is vLLM (in-process or HTTP).

    HuggingFace packed decode is **not** selected automatically. Set
    ``SAE_GENERATE_BACKEND=hf`` and call ``generate_hf_packed`` from the task
    if you must use the slow path.
    """
    if not prompts:
        return []
    if resolve_backend() == "hf":
        raise RuntimeError(
            "saebase.generate.generate() does not run HuggingFace decode. "
            "Use an explicit Hugging Face generation implementation or unset "
            "SAE_GENERATE_BACKEND to select vLLM."
        )
    if vllm_available() or llm is not None:
        engine = llm if llm is not None else _cached_llm(
            model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        return generate_vllm(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            on_complete=on_complete,
            llm=engine,
            seed=seed,
        )
    from saebase.generate.server import server_base_url, server_healthy

    if server_healthy():
        return _rows_from_http(
            prompts,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            on_complete=on_complete,
            seed=seed,
        )
    raise RuntimeError(
        "vLLM is not in this interpreter and no server is listening at "
        f"{server_base_url()}. Install the optional stack via "
        "pip install -e '.[generation]' or explicitly configure a vLLM HTTP server."
    )
