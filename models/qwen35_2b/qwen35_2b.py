"""Qwen3.5 2B Base with registered public TopK residual SAEs."""
from __future__ import annotations

from typing import Any
from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec
from saebase.data.prompt_schema import behaviors_for_model

SPEC = BaseModelSpec(
    key="qwen35_2b",
    model_name="Qwen/Qwen3.5-2B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3.5-2B-Base",),
    notes="Dense 2B Qwen3.5 hybrid DeltaNet/full attention. Official W32K TopK-50 SAEs.",
)
SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen35_2b")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    """Legacy TransformerLens integration. Stage A validates with HF layer hooks."""
    return load_from_spec(SPEC, device=device)


def construct(cfg: DictConfig) -> HookedLMHandle:
    return load_model(device=str(cfg.model.get("device", "cuda")))


def construct_lazy(cfg: DictConfig) -> dict[str, Any]:
    return {
        "key": SPEC.key, "model_name": SPEC.model_name, "family": SPEC.family,
        "lazy": True, "device": str(cfg.model.get("device", "cuda")),
    }
