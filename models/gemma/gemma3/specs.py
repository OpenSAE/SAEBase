"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

gemma1b_SPEC = BaseModelSpec(
    key="gemma1b",
    model_name="google/gemma-3-1b-pt",
    family="gemma",
    model_candidates=("google/gemma-3-1b-pt", "gemma-3-1b"),
    notes="Gemma-3 1B PT; residual SAEs via Gemma Scope 2 (16k).",
)
gemma1b = ModelAdapter(
    gemma1b_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma1b_it_SPEC = BaseModelSpec(
    key="gemma1b_it",
    model_name="google/gemma-3-1b-it",
    family="gemma",
    model_candidates=("google/gemma-3-1b-it", "gemma-3-1b-it", "unsloth/gemma-3-1b-it"),
    notes="Gemma-3 1B Instruct; P0 aligned pair with gemma1b (PT). May load unsloth/gemma-3-1b-it if google/ is gated; record weight_source_deviation.",
)
gemma1b_it = ModelAdapter(
    gemma1b_it_SPEC,
    backend="load_from_spec",
    options={"architecture_aliases": {"unsloth/gemma-3-1b-it": "google/gemma-3-1b-it"}},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma270m_SPEC = BaseModelSpec(
    key="gemma270m",
    model_name="gemma-3-270m",
    family="gemma",
    model_candidates=("gemma-3-270m", "google/gemma-3-270m"),
    notes="Gemma-3 270M; residual SAEs via Gemma Scope 2.",
)
gemma270m = ModelAdapter(
    gemma270m_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma3_12b_it_SPEC = BaseModelSpec(
    key="gemma3_12b_it",
    model_name="google/gemma-3-12b-it",
    family="gemma",
    model_candidates=("google/gemma-3-12b-it",),
    notes="Gemma-3 12B IT; native Scope 2 residual SAE.",
)
gemma3_12b_it = ModelAdapter(
    gemma3_12b_it_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma3_12b_pt_SPEC = BaseModelSpec(
    key="gemma3_12b_pt",
    model_name="google/gemma-3-12b-pt",
    family="gemma",
    model_candidates=("google/gemma-3-12b-pt",),
    notes="Gemma-3 12B PT; Scope 2 resid_post 65k medium at official four depths.",
)
gemma3_12b_pt = ModelAdapter(
    gemma3_12b_pt_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma3_27b_it_SPEC = BaseModelSpec(
    key="gemma3_27b_it",
    model_name="google/gemma-3-27b-it",
    family="gemma",
    model_candidates=("google/gemma-3-27b-it",),
    notes="Gemma-3 27B IT; native Scope 2 residual SAE.",
)
gemma3_27b_it = ModelAdapter(
    gemma3_27b_it_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma3_27b_pt_SPEC = BaseModelSpec(
    key="gemma3_27b_pt",
    model_name="google/gemma-3-27b-pt",
    family="gemma",
    model_candidates=("google/gemma-3-27b-pt",),
    notes="Gemma-3 27B PT; Scope 2 resid_post 65k medium at official four depths.",
)
gemma3_27b_pt = ModelAdapter(
    gemma3_27b_pt_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma3_4b_it_SPEC = BaseModelSpec(
    key="gemma3_4b_it",
    model_name="google/gemma-3-4b-it",
    family="gemma",
    model_candidates=("google/gemma-3-4b-it",),
    notes="Gemma-3 4B IT; native Scope 2 residual SAE (same layers as PT).",
)
gemma3_4b_it = ModelAdapter(
    gemma3_4b_it_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


gemma3_4b_pt_SPEC = BaseModelSpec(
    key="gemma3_4b_pt",
    model_name="google/gemma-3-4b-pt",
    family="gemma",
    model_candidates=("google/gemma-3-4b-pt",),
    notes="Gemma-3 4B PT; Scope 2 resid_post 65k medium at official 25/50/65/85% layers.",
)
gemma3_4b_pt = ModelAdapter(
    gemma3_4b_pt_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
