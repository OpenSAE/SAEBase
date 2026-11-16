"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
import torch
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

mistral7b_instruct_SPEC = BaseModelSpec(
    key="mistral7b_instruct",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    family="mistral",
    model_candidates=("mistralai/Mistral-7B-Instruct-v0.1",),
    notes="SAE at layer-16 resid_pre; 131K TopK features from tylercosgrove. Loaded in bfloat16 so model+131k SAE fit on ~32GB GPUs.",
)
mistral7b_instruct = ModelAdapter(
    mistral7b_instruct_SPEC,
    backend="load_from_spec",
    options={"dtype": torch.bfloat16},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
