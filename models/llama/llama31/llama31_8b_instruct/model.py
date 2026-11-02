"""Llama-3.1-8B-Instruct HookedTransformer model (protocol 10)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

from saebase.datasets.prompt_schema import behaviors_for_model

SPEC = BaseModelSpec(
    key="llama31_8b_instruct",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    family="llama",
    model_candidates=("meta-llama/Llama-3.1-8B-Instruct",),
    notes="Llama-3.1 8B Instruct; andyrdt BatchTopK residual SAE L19 k=64.",
)


SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("llama31_8b_instruct")


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
