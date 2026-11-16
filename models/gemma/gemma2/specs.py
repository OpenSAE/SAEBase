"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

gemma2_2b_base_SPEC = BaseModelSpec(
    key="gemma2_2b_base",
    model_name="google/gemma-2-2b",
    family="gemma2",
    model_candidates=("google/gemma-2-2b", "gemma-2-2b"),
    notes="Gemma-2 2B PT; P0 primary ModelPair base; final layer = 25.",
)
gemma2_2b_base = ModelAdapter(
    gemma2_2b_base_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma2_2b_it_SPEC = BaseModelSpec(
    key="gemma2_2b_it",
    model_name="google/gemma-2-2b-it",
    family="gemma2",
    model_candidates=("google/gemma-2-2b-it", "gemma-2-2b-it"),
    notes="Gemma-2 2B Instruct; P0 primary ModelPair instruct; final layer = 25.",
)
gemma2_2b_it = ModelAdapter(
    gemma2_2b_it_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma2_9b_it_SPEC = BaseModelSpec(
    key="gemma2_9b_it",
    model_name="google/gemma-2-9b-it",
    family="gemma",
    model_candidates=("google/gemma-2-9b-it",),
    notes="Gemma-2 9B IT; native Scope residual at layers 9/20/31.",
)
gemma2_9b_it = ModelAdapter(
    gemma2_9b_it_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma2_9b_pt_SPEC = BaseModelSpec(
    key="gemma2_9b_pt",
    model_name="google/gemma-2-9b",
    family="gemma",
    model_candidates=("google/gemma-2-9b",),
    notes="Gemma-2 9B PT; Scope 16k residual at layers 9/20/31 (PT/IT aligned).",
)
gemma2_9b_pt = ModelAdapter(
    gemma2_9b_pt_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
