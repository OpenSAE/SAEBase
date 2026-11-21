"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from pathlib import Path
import torch
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

qwen35_27b_SPEC = BaseModelSpec(
    key="qwen35_27b",
    model_name="Qwen/Qwen3.5-27B",
    family="qwen",
    model_candidates=("Qwen/Qwen3.5-27B",),
    notes="Qwen3.5 27B; official residual TopK SAE W80K L0_50.",
)
qwen35_27b = ModelAdapter(
    qwen35_27b_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


qwen35_2b_SPEC = BaseModelSpec(
    key="qwen35_2b",
    model_name="Qwen/Qwen3.5-2B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3.5-2B-Base",),
    notes="Official Qwen3.5 2B base with residual TopK SAE W32K K50.",
)
qwen35_2b = ModelAdapter(
    qwen35_2b_SPEC,
    backend="load_hf_hooks_model",
    options={"dtype": torch.bfloat16},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


qwen35_4b__HUB = "Qwen/Qwen3.5-4B"
qwen35_4b__LOCAL = "/workspace/data/models/Qwen3.5-4B"
qwen35_4b__candidates = (qwen35_4b__HUB,)
if Path(qwen35_4b__LOCAL).joinpath("config.json").is_file():
    qwen35_4b__candidates = (qwen35_4b__LOCAL, qwen35_4b__HUB)
qwen35_4b_SPEC = BaseModelSpec(
    key="qwen35_4b",
    model_name=qwen35_4b__HUB,
    family="qwen",
    model_candidates=qwen35_4b__candidates,
    notes="Qwen3.5 4B post-trained/chat; Experiment 11d host with decoderesearch Matryoshka BatchTopK residual SAE at L15 (65k, training k=100). Uses native HF residual hooks because TransformerLens does not yet register this hybrid architecture.",
)
qwen35_4b = ModelAdapter(
    qwen35_4b_SPEC,
    backend="load_hf_hooks_model",
    options={"dtype": torch.bfloat16},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


qwen35_9b_SPEC = BaseModelSpec(
    key="qwen35_9b",
    model_name="Qwen/Qwen3.5-9B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3.5-9B-Base",),
    notes="Qwen3.5 9B base; official residual TopK SAE W64K L0_50. Uses native HF residual hooks for the hybrid Qwen3.5 architecture.",
)
qwen35_9b = ModelAdapter(
    qwen35_9b_SPEC,
    backend="load_hf_hooks_model",
    options={"dtype": torch.bfloat16},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
