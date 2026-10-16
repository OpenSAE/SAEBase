"""Mistral-7B-Instruct-v0.1, paired with a community layer-16 SAE."""
from __future__ import annotations

from typing import Any
from omegaconf import DictConfig
from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="mistral7b_instruct",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    family="mistral",
    model_candidates=("mistralai/Mistral-7B-Instruct-v0.1",),
    notes="SAE at layer-16 resid_pre; 131K TopK features from tylercosgrove.",
)

def get_spec() -> BaseModelSpec:
    return SPEC

def load_model(device: str | None = None) -> HookedLMHandle:
    return load_from_spec(SPEC, device=device)

def construct(cfg: DictConfig) -> HookedLMHandle:
    return load_model(device=str(cfg.model.device) if cfg.model.get("device") is not None else None)

def construct_lazy(cfg: DictConfig) -> dict[str, Any]:
    return {"key":SPEC.key, "model_name":SPEC.model_name, "family":SPEC.family,
            "lazy":True, "device":str(cfg.model.get("device","cuda"))}
