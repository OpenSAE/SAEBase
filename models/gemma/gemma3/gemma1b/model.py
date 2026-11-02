"""Gemma-3 1B PT HookedTransformer model."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

from saebase.datasets.prompt_schema import behaviors_for_model

SPEC = BaseModelSpec(
    key="gemma1b",
    model_name="google/gemma-3-1b-pt",
    family="gemma",
    model_candidates=("google/gemma-3-1b-pt", "gemma-3-1b"),
    notes="Gemma-3 1B PT; residual SAEs via Gemma Scope 2 (16k).",
)


SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("gemma1b")


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
