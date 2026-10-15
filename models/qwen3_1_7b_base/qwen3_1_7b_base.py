"""Qwen3-1.7B-Base model identity for official Qwen-Scope SAE weights."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="qwen3_1_7b_base",
    model_name="Qwen/Qwen3-1.7B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3-1.7B-Base",),
    notes="Official Qwen3-1.7B Base with Qwen-Scope Residual TopK-50 32K.",
)

def get_spec() -> BaseModelSpec:
    return SPEC

def load_model(device: str | None = None) -> HookedLMHandle:
    return load_from_spec(SPEC, device=device)

def construct(cfg: DictConfig) -> HookedLMHandle:
    return load_model(device=str(cfg.model.device) if cfg.model.get("device") is not None else None)

def construct_lazy(cfg: DictConfig) -> dict[str, Any]:
    return {"key": SPEC.key, "model_name": SPEC.model_name, "family": SPEC.family,
            "lazy": True, "device": str(cfg.model.get("device", "cuda"))}
