"""Qwen2.5-3B-Instruct model (HF logits backend)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hf_logits_backend import load_hf_logits_model
from models.hooked_lm_common import BaseModelSpec, HookedLMHandle

SPEC = BaseModelSpec(
    key="qwen25_3b_it",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    family="qwen",
    model_candidates=("Qwen/Qwen2.5-3B-Instruct",),
    notes="Qwen2.5 3B Instruct; HF logits backend; P0 aligned pair with qwen25_3b.",
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("qwen25_3b_it")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    return load_hf_logits_model(
        SPEC.model_name,
        key=SPEC.key,
        family=SPEC.family,
        device=device or "cuda",
    )


def construct(cfg: DictConfig) -> HookedLMHandle:
    dev = None
    if cfg.model.get("device") is not None:
        dev = str(cfg.model.device)
    return load_model(device=dev)


def construct_lazy(cfg: DictConfig) -> dict[str, Any]:
    return {
        "key": SPEC.key,
        "model_name": SPEC.model_name,
        "family": SPEC.family,
        "lazy": True,
        "device": str(cfg.model.get("device", "cuda")),
        "backend": "hf",
    }
