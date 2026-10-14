"""Llama-3.1-8B HookedTransformer model."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="llama31_8b",
    model_name="meta-llama/Llama-3.1-8B",
    family="llama",
    model_candidates=("meta-llama/Llama-3.1-8B",),
    notes="Llama-3.1 8B base; Llama-Scope LXR-8x residual (TopK≈50).",
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("llama31_8b")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    return load_from_spec(SPEC, device=device)


def construct(cfg: DictConfig) -> HookedLMHandle:
    device = None
    if cfg.model.get("device") is not None:
        device = str(cfg.model.device)
    return load_model(device=device)


def construct_lazy(cfg: DictConfig) -> dict[str, Any]:
    return {
        "key": SPEC.key,
        "model_name": SPEC.model_name,
        "family": SPEC.family,
        "lazy": True,
        "device": str(cfg.model.get("device", "cuda")),
    }
