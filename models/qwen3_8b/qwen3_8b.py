"""Qwen3-8B-Base HookedTransformer model."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="qwen3_8b",
    model_name="Qwen/Qwen3-8B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3-8B-Base",),
    notes="Qwen3 8B base; official residual TopK SAE W64K L0_50.",
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen3_8b")


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
