"""Qwen3-8B-Base model identity for Stage D residual SAE coverage."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from models.hf_hooks_backend import load_hf_hooks_model
from models.hooked_lm_common import BaseModelSpec, HookedLMHandle

SPEC = BaseModelSpec(
    key="qwen3_8b",
    model_name="Qwen/Qwen3-8B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3-8B-Base",),
    notes=(
        "Qwen3 8B base; official residual TopK SAE W64K L0_50. "
        "Loaded via native HF residual hooks for TransformerLens gaps."
    ),
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen3_8b")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    return load_hf_hooks_model(
        SPEC.model_name,
        key=SPEC.key,
        family=SPEC.family,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
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
