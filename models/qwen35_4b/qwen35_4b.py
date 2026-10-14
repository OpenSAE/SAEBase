"""Qwen3.5-4B HookedTransformer model (Experiment 11d host)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

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
        "(65k, training k=100)."
    ),
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen35_4b")


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
