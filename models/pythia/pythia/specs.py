"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

pythia410m_SPEC = BaseModelSpec(
    key="pythia410m",
    model_name="pythia-410m-deduped",
    family="pythia",
    notes="EleutherAI Pythia-410M; MLP TopK SAEs (not residual sae_lens).",
)
pythia410m = ModelAdapter(
    pythia410m_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


pythia70m_SPEC = BaseModelSpec(
    key="pythia70m",
    model_name="pythia-70m-deduped",
    family="pythia",
    notes="EleutherAI Pythia-70M deduped; residual SAEs via sae_lens.",
)
pythia70m = ModelAdapter(
    pythia70m_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
