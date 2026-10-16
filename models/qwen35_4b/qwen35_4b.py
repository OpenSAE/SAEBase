"""Qwen3.5-4B model identity for Stage D Matryoshka SAE coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from models.hf_hooks_backend import load_hf_hooks_model
from models.hooked_lm_common import BaseModelSpec, HookedLMHandle

_HUB = "Qwen/Qwen3.5-4B"
_LOCAL = "/workspace/data/models/Qwen3.5-4B"

_candidates = (_HUB,)
if Path(_LOCAL).joinpath("config.json").is_file():
    _candidates = (_LOCAL, _HUB)

SPEC = BaseModelSpec(
    key="qwen35_4b",
    model_name=_HUB,
    family="qwen",
    model_candidates=_candidates,
    notes=(
        "Qwen3.5 4B post-trained/chat; Experiment 11d host with "
        "decoderesearch Matryoshka BatchTopK residual SAE at L15 "
        "(65k, training k=100). Uses native HF residual hooks because "
        "TransformerLens does not yet register this hybrid architecture."
    ),
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen35_4b")


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
