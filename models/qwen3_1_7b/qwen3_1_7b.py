"""Qwen3-1.7B HookedTransformer model (Experiment 11b host)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

_HUB = "Qwen/Qwen3-1.7B"
_LOCAL = "/workspace/data/models/Qwen3-1.7B"

_candidates = (_HUB,)
if Path(_LOCAL).joinpath("config.json").is_file():
    _candidates = (_LOCAL, _HUB)

SPEC = BaseModelSpec(
    key="qwen3_1_7b",
    model_name=_HUB,
    family="qwen",
    model_candidates=_candidates,
    notes=(
        "Qwen3 1.7B instruct/chat; Experiment 11b host with adamkarvonen "
        "BatchTopK residual SAE at L14 (65k, L0≈80)."
    ),
)

from saebase.datasets.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen3_1_7b")


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
