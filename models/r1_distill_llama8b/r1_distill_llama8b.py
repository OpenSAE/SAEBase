"""DeepSeek-R1-Distill-Llama-8B HookedTransformer model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec

_LOCAL = "/workspace/data/models/DeepSeek-R1-Distill-Llama-8B"
_HUB = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
_ARCH = "meta-llama/Llama-3.1-8B"

_candidates = (_HUB,)
if Path(_LOCAL).joinpath("config.json").is_file():
    _candidates = (_LOCAL, _HUB)

SPEC = BaseModelSpec(
    key="r1_distill_llama8b",
    model_name=_HUB,
    family="llama",
    model_candidates=_candidates,
    notes=(
        "R1-distilled Llama-3.1-8B; 02a Step-1 host. "
        "TransformerLens has no official id, so weights convert through Llama-3.1-8B."
    ),
)

from saebase.data.prompt_schema import behaviors_for_model

SUPPORTED_BEHAVIORS: frozenset[str] = behaviors_for_model("r1_distill_llama8b")


def get_spec() -> BaseModelSpec:
    return SPEC


def load_model(device: str | None = None) -> HookedLMHandle:
    aliases = {_HUB: _ARCH}
    if Path(_LOCAL).joinpath("config.json").is_file():
        aliases[_LOCAL] = _ARCH
    return load_from_spec(SPEC, device=device, architecture_aliases=aliases)


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
