"""Allowlisted local explanation generator.

Hydra YAML cannot store a Python callable. ``task.generator=saebase_generate``
selects this function. It calls ``saebase.generate.generate`` and does not eval
configuration text, import vLLM directly, or call an external paid API.

``heldout_score`` is computed by the caller from SAE activations. This function
only returns candidate text.
"""

from __future__ import annotations

from collections.abc import Callable

from tasks.sae_runtime import _task


def saebase_generate(cfg: object) -> Callable[[list[str]], str]:
    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Auto-explanation needs task.approved=true")
    max_new = _task(cfg, "generator_max_new_tokens", 24)
    if type(max_new) is not int or not 1 <= max_new <= 64:
        raise ValueError("generator_max_new_tokens must be an integer from 1 to 64")
    model = str(_task(cfg, "generator_model", "") or "").strip()
    if not model:
        raise ValueError("task.generator_model must name the local or cached model")
    dtype = str(_task(cfg, "generator_dtype", "bfloat16"))
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("generator_dtype must be float32, float16, or bfloat16")

    def generator(snippets: list[str]) -> str:
        if not snippets or any(not isinstance(text, str) or not text for text in snippets):
            raise ValueError("Generator needs the activating discovery snippets")
        ask = "In one short sentence, what do these snippets have in common? " + " | ".join(
            snippets
        )
        if len(ask) > 4000:
            raise ValueError("Explanation prompt exceeds 4000 characters")
        from saebase.generate import generate

        rows = generate(
            [ask],
            max_new_tokens=max_new,
            temperature=0.0,
            top_p=1.0,
            model=model,
            gpu_memory_utilization=float(_task(cfg, "gpu_memory_utilization", 0.8)),
            max_model_len=int(_task(cfg, "max_model_len", 256)),
            dtype=dtype,
            seed=0,
        )
        text = str(rows[0].get("text") or "").strip()
        if not text:
            raise RuntimeError("saebase.generate returned no text")
        return text[:800]

    return generator
