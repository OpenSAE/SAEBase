"""Series model identities and explicit loading and compatibility declarations."""

from __future__ import annotations
from pathlib import Path
import torch
from models.hooked_lm_common import BaseModelSpec
from models.model_adapter import ModelAdapter

qwen3_1_7b__HUB = "Qwen/Qwen3-1.7B"
qwen3_1_7b__LOCAL = "/workspace/data/models/Qwen3-1.7B"
qwen3_1_7b__candidates = (qwen3_1_7b__HUB,)
if Path(qwen3_1_7b__LOCAL).joinpath("config.json").is_file():
    qwen3_1_7b__candidates = (qwen3_1_7b__LOCAL, qwen3_1_7b__HUB)
qwen3_1_7b_SPEC = BaseModelSpec(
    key="qwen3_1_7b",
    model_name=qwen3_1_7b__HUB,
    family="qwen",
    model_candidates=qwen3_1_7b__candidates,
    notes="Qwen3 1.7B instruct/chat; Experiment 11b host with adamkarvonen BatchTopK residual SAE at L14 (65k, L0≈80).",
)
qwen3_1_7b = ModelAdapter(
    qwen3_1_7b_SPEC,
    backend="load_from_spec",
    options={},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


qwen3_1_7b_base_SPEC = BaseModelSpec(
    key="qwen3_1_7b_base",
    model_name="Qwen/Qwen3-1.7B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3-1.7B-Base",),
    notes="Official Qwen3-1.7B Base with Qwen-Scope Residual TopK-50 32K.",
)
qwen3_1_7b_base = ModelAdapter(
    qwen3_1_7b_base_SPEC,
    backend="load_hf_hooks_model",
    options={"dtype": torch.bfloat16},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)


qwen3_8b_SPEC = BaseModelSpec(
    key="qwen3_8b",
    model_name="Qwen/Qwen3-8B-Base",
    family="qwen",
    model_candidates=("Qwen/Qwen3-8B-Base",),
    notes="Qwen3 8B base; official residual TopK SAE W64K L0_50. Loaded via native HF residual hooks for TransformerLens gaps.",
)
qwen3_8b = ModelAdapter(
    qwen3_8b_SPEC,
    backend="load_hf_hooks_model",
    options={"dtype": torch.bfloat16},
    construct_default_cuda=False,
    lazy_backend=None,
    local_alias=None,
)
