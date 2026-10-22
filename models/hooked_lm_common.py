"""Version-aware TransformerLens loading helpers for SAE / SAR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


# Canonical prompt-behavior names for SAEBase evaluations.
TEXT_BEHAVIORS: frozenset[str] = frozenset(
    {
        "CounterFact",
        "IOI",
        "GreaterThan",
        "SubjectVerbAgreement",
        "SimpleCopying",
        "MMLUPro",
    }
)


@dataclass(frozen=True)
class BaseModelSpec:
    """Stable identity for one HookedTransformer base model (no SAE sites)."""

    key: str
    model_name: str
    family: str
    model_candidates: tuple[str, ...] = ()
    notes: str = ""

    def candidates(self) -> tuple[str, ...]:
        if self.model_candidates:
            return self.model_candidates
        return (self.model_name,)


@dataclass(frozen=True)
class HookedLMHandle:
    """Lightweight handle returned by hooked LM ``construct`` plugins."""

    model: Any
    loaded_name: str
    key: str
    family: str


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hooked LM models require CUDA; torch.cuda.is_available() is False."
        )
    return "cuda"


def _default_load_kwargs(model_name: str) -> dict[str, Any]:
    """Memory-safe defaults for Gemma loads on modest host RAM / VRAM.

    ``from_pretrained`` post-processing for gemma-3-1b can OOM ~15GB hosts;
    skip processing but keep float32 so logit filtering stays finite.
    Gemma-2-2B uses float16 to fit 16GB with SAE headroom.
    Larger Gemma-3 / Llama / Qwen checkpoints default to bfloat16.
    Quantized HF checkpoints are allowed when VRAM is tight (8-bit / 4-bit).
    """
    name = model_name.lower()
    if "gemma-3-1b" in name:
        return {"no_processing": True, "dtype": torch.float32}
    if "qwen2.5-1.5b" in name or "deepseek-r1-distill-qwen-1.5b" in name:
        return {"dtype": torch.float32}
    if "gemma-2-2b" in name:
        return {"dtype": torch.float16}
    if any(
        tok in name
        for tok in (
            "gemma-3-4b",
            "gemma-3-12b",
            "gemma-3-27b",
            "gemma-2-9b",
            "llama-3.1-8b",
            "deepseek-r1-distill-llama-8b",
            "qwen3",
            "qwen3.5",
            "mistral-7b",
        )
    ):
        return {"no_processing": True, "dtype": torch.bfloat16}
    return {}


def load_hooked_model(
    model_name: str,
    device: str | None = None,
    *,
    dtype: torch.dtype | None = None,
    no_processing: bool | None = None,
    hf_model: Any | None = None,
    tokenizer: Any | None = None,
    architecture_name: str | None = None,
) -> Any:
    from models.transformerlens_backend import boot_transformerlens

    dev = device or require_cuda()
    defaults = _default_load_kwargs(architecture_name or model_name)
    use_no_processing = (
        defaults.get("no_processing", False)
        if no_processing is None
        else bool(no_processing)
    )
    use_dtype = defaults.get("dtype") if dtype is None else dtype
    # Architecture name must be recognized by the selected TL model system;
    # hf_model/tokenizer may refer to the weights of a separate source mirror.
    load_name = architecture_name or model_name
    model, _ = boot_transformerlens(
        load_name, device=dev, dtype=use_dtype,
        no_processing=use_no_processing, hf_model=hf_model, tokenizer=tokenizer,
    )
    return model


def load_hf_causal_lm(repo_id: str, *, dtype: torch.dtype | None = None) -> tuple[Any, Any]:
    """Load HF CausalLM + tokenizer on CPU. Quantized checkpoints are allowed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    # Keep HF weights on CPU during conversion to avoid doubling VRAM with TL.
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=dtype or torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    return model, tok


def load_hooked_model_with_candidates(
    model_names: tuple[str, ...] | list[str],
    device: str | None = None,
    *,
    dtype: torch.dtype | None = None,
    no_processing: bool | None = None,
    architecture_aliases: dict[str, str] | None = None,
) -> tuple[Any, str]:
    """Try each candidate name; return (model, loaded_name).

    ``architecture_aliases`` maps a weight-repo candidate (e.g. unsloth/...)
    to a TransformerLens official architecture id (e.g. google/gemma-3-1b-it).
    """
    import gc

    errors: list[str] = []
    aliases = architecture_aliases or {}
    for name in model_names:
        hf_model = None
        try:
            if name in aliases:
                # Load weights from mirror, structure from official TL name.
                hf_dtype = dtype or _default_load_kwargs(aliases[name]).get("dtype")
                hf_model, tok = load_hf_causal_lm(name, dtype=hf_dtype)
                model = load_hooked_model(
                    name,
                    device=device,
                    dtype=dtype,
                    no_processing=no_processing,
                    hf_model=hf_model,
                    tokenizer=tok,
                    architecture_name=aliases[name],
                )
                # Drop HF copy ASAP (TL already converted state).
                del hf_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return model, name
            return (
                load_hooked_model(
                    name,
                    device=device,
                    dtype=dtype,
                    no_processing=no_processing,
                ),
                name,
            )
        except Exception as exc:  # noqa: BLE001 — try next candidate
            if hf_model is not None:
                del hf_model
                gc.collect()
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError(
        "Could not load any model candidate.\n" + "\n".join(errors)
    )


def load_from_spec(
    spec: BaseModelSpec,
    device: str | None = None,
    *,
    dtype: torch.dtype | None = None,
    no_processing: bool | None = None,
    architecture_aliases: dict[str, str] | None = None,
) -> HookedLMHandle:
    model, loaded = load_hooked_model_with_candidates(
        spec.candidates(),
        device=device,
        dtype=dtype,
        no_processing=no_processing,
        architecture_aliases=architecture_aliases,
    )
    # Attach source metadata for P0-D1 audit (non-invasive attribute).
    try:
        model._saebase_weight_source = loaded  # type: ignore[attr-defined]
        model._saebase_weight_source_deviation = loaded != spec.model_name  # type: ignore[attr-defined]
    except Exception:
        pass
    return HookedLMHandle(
        model=model,
        loaded_name=loaded,
        key=spec.key,
        family=spec.family,
    )


def model_forward_smoke(model: Any, text: str = "Hello world") -> tuple[int, ...]:
    out = model(text)
    return tuple(out.shape)


def print_vram_gb(label: str = "VRAM peak (GB)") -> None:
    if not torch.cuda.is_available():
        print(f"{label}: N/A (CPU)")
        return
    gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label}: {gb:.3f}")
