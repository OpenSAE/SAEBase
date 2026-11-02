"""Pythia-410M deduped HookedTransformer model."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

from saebase.datasets.prompt_schema import behaviors_for_model

SPEC = BaseModelSpec(
    key="pythia410m",
    model_name="pythia-410m-deduped",
    family="pythia",
    notes="EleutherAI Pythia-410M; MLP TopK SAEs (not residual sae_lens).",
)


SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("pythia410m")


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
