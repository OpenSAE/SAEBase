"""DeepSeek-R1-Distill-Qwen-1.5B HookedTransformer model."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

SPEC = BaseModelSpec(
    key="r1_distill_qwen15b",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    family="qwen",
    model_candidates=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",),
    notes=(
        "R1-distilled Qwen2.5-Math-1.5B; first-round ordered-division PoC. "
        "Classic TransformerLens official names do not include this Hub id, "
        "so weights are converted through the Qwen2.5-1.5B architecture."
    ),
)

from saebase.datasets.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("r1_distill_qwen15b")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    return load_from_spec(
        SPEC,
        device=device,
        architecture_aliases={
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Qwen/Qwen2.5-1.5B",
        },
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
