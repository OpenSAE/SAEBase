"""task × prompt_source compatibility for ``llama31_8b_instruct``."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.datasets.prompt_schema import behaviors_for_model
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in

_TASKS: FrozenSet[str] = frozenset({"sae", "vllm_sae_steer", "cross_model_filtering", "sae_upgrade"})

_ALIAS = {
    "CounterFact": "counterfact",
    "IOI": "ioi",
    "GreaterThan": "greater_than",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "SimpleCopying": "simple_copying",
    "MMLUPro": "mmlu_pro",
    "BBH": "bbh",
    "ARCChallenge": "arc_challenge",
    "MedQA": "medqa",
    "MATH500": "math500",
    "GPQAExtended": "gpqa_extended",
    "LiveCodeBench": "livecodebench",
}


def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, _TASKS, "llama31_8b_instruct")
    allowed = set(behaviors_for_model("llama31_8b_instruct"))
    for beh in list(allowed):
        if beh in _ALIAS:
            allowed.add(_ALIAS[beh])
    assert_prompt_source_in(cfg, frozenset(allowed), "llama31_8b_instruct")
