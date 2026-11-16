"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

llama31_8b_SPEC = BaseModelSpec(
    key="llama31_8b",
    model_name="meta-llama/Llama-3.1-8B",
    family="llama",
    model_candidates=("meta-llama/Llama-3.1-8B",),
    notes="Llama-3.1 8B base; Llama-Scope LXR-8x residual (TopK≈50).",
)
llama31_8b = ModelAdapter(
    llama31_8b_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


llama31_8b_instruct_SPEC = BaseModelSpec(
    key="llama31_8b_instruct",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    family="llama",
    model_candidates=("meta-llama/Llama-3.1-8B-Instruct",),
    notes="Llama-3.1 8B Instruct; andyrdt BatchTopK residual SAE L19 k=64.",
)
llama31_8b_instruct = ModelAdapter(
    llama31_8b_instruct_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
