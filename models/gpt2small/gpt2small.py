"""GPT-2 Small HookedTransformer model for M5 ungated public release matrix."""
from __future__ import annotations
from typing import Any
from omegaconf import DictConfig
from models.hooked_lm_common import BaseModelSpec, HookedLMHandle, load_from_spec
from saebase.datasets.prompt_schema import behaviors_for_model

SPEC=BaseModelSpec(
    key="gpt2small", model_name="gpt2-small", family="gpt2",
    model_candidates=("gpt2-small",),
    notes="OpenAI GPT-2 Small; public MIT model and layer-8 residual SAE families.",
)
SUPPORTED_BEHAVIORS=frozenset(behaviors_for_model("gpt2small"))

def get_spec()->BaseModelSpec:
    return SPEC

def load_model(device: str|None=None)->HookedLMHandle:
    return load_from_spec(SPEC,device=device)

def construct(cfg: DictConfig)->HookedLMHandle:
    return load_model(device=str(cfg.model.get("device","cuda")))

def construct_lazy(cfg: DictConfig)->dict[str,Any]:
    return {"key":SPEC.key,"model_name":SPEC.model_name,
            "family":SPEC.family,"lazy":True,
            "device":str(cfg.model.get("device","cuda"))}
