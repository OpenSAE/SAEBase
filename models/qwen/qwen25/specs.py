"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

qwen25_3b_SPEC = BaseModelSpec(
    key="qwen25_3b",
    model_name="Qwen/Qwen2.5-3B",
    family="qwen",
    model_candidates=("Qwen/Qwen2.5-3B",),
    notes="Qwen2.5 3B base; HF logits backend for R3 D4 cross-model contrast.",
)
qwen25_3b = ModelAdapter(
    qwen25_3b_SPEC,
    backend="load_hf_logits_model",
    options={},
    construct_default_cuda=False,
    lazy_backend="hf",
    local_alias=None,
)


qwen25_3b_it_SPEC = BaseModelSpec(
    key="qwen25_3b_it",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    family="qwen",
    model_candidates=("Qwen/Qwen2.5-3B-Instruct",),
    notes="Qwen2.5 3B Instruct; HF logits backend; P0 aligned pair with qwen25_3b.",
)
qwen25_3b_it = ModelAdapter(
    qwen25_3b_it_SPEC,
    backend="load_hf_logits_model",
    options={},
    construct_default_cuda=False,
    lazy_backend="hf",
    local_alias=None,
)
