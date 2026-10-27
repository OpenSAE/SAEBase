"""Gemma-3 1B Instruct HookedTransformer model."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="gemma1b_it",
    model_name="google/gemma-3-1b-it",
    family="gemma",
    model_candidates=(
        "google/gemma-3-1b-it",
        "gemma-3-1b-it",
        # Full-precision public compatible mirror when official repo is gated.
        "unsloth/gemma-3-1b-it",
    ),
    notes=(
        "Gemma-3 1B Instruct; P0 aligned pair with gemma1b (PT). "
        "May load unsloth/gemma-3-1b-it if google/ is gated; record weight_source_deviation."
    ),
)

from saebase.datasets.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("gemma1b_it")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    # Official google/* may be gated; unsloth is full-precision compatible weights
    # but TransformerLens only knows the google architecture name → alias map.
    return load_from_spec(
        SPEC,
        device=device,
        architecture_aliases={
            "unsloth/gemma-3-1b-it": "google/gemma-3-1b-it",
        },
    )


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
