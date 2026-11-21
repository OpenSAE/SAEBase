"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

r1_distill_qwen15b_SPEC = BaseModelSpec(
    key="r1_distill_qwen15b",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    family="qwen",
    model_candidates=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",),
    notes="R1-distilled Qwen2.5-Math-1.5B; first-round ordered-division PoC. Classic TransformerLens official names do not include this Hub id, so weights are converted through the Qwen2.5-1.5B architecture.",
)
r1_distill_qwen15b = ModelAdapter(
    r1_distill_qwen15b_SPEC,
    backend="load_from_spec",
    options={
        "architecture_aliases": {"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Qwen/Qwen2.5-1.5B"}
    },
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
