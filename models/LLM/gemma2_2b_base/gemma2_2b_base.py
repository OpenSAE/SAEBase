"""Gemma-2 2B PT HookedTransformer model (P0 primary base)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="gemma2_2b_base",
    model_name="google/gemma-2-2b",
    family="gemma2",
    model_candidates=("google/gemma-2-2b", "gemma-2-2b"),
    notes="Gemma-2 2B PT; P0 primary ModelPair base; final layer = 25.",
)

from research_datasets.NLP.artifacts import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("gemma2_2b_base")


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
