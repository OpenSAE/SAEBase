"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from pathlib import Path
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

r1_distill_llama8b__LOCAL = "/workspace/data/models/DeepSeek-R1-Distill-Llama-8B"
r1_distill_llama8b__HUB = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
r1_distill_llama8b__ARCH = "meta-llama/Llama-3.1-8B"
r1_distill_llama8b__candidates = (r1_distill_llama8b__HUB,)
if Path(r1_distill_llama8b__LOCAL).joinpath("config.json").is_file():
    r1_distill_llama8b__candidates = (r1_distill_llama8b__LOCAL, r1_distill_llama8b__HUB)
r1_distill_llama8b_SPEC = BaseModelSpec(
    key="r1_distill_llama8b",
    model_name=r1_distill_llama8b__HUB,
    family="llama",
    model_candidates=r1_distill_llama8b__candidates,
    notes="R1-distilled Llama-3.1-8B; 02a Step-1 host. TransformerLens has no official id, so weights convert through Llama-3.1-8B.",
)
r1_distill_llama8b = ModelAdapter(
    r1_distill_llama8b_SPEC,
    backend="load_from_spec",
    options={"architecture_aliases": {r1_distill_llama8b__HUB: r1_distill_llama8b__ARCH}},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=(r1_distill_llama8b__LOCAL, r1_distill_llama8b__ARCH),
)
