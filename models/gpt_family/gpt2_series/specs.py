"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

gpt2small_SPEC = BaseModelSpec(
    key="gpt2small",
    model_name="gpt2-small",
    family="gpt2",
    model_candidates=("gpt2-small",),
    notes="OpenAI GPT-2 Small; public MIT model and layer-8 residual SAE families.",
)
gpt2small = ModelAdapter(
    gpt2small_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=True,
    lazy_backend=None,
    local_alias=None,
)
