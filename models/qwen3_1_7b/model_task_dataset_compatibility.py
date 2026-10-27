"""task × prompt_source compatibility for ``qwen3_1_7b``."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.datasets.prompt_schema import behaviors_for_model
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in

_TASKS: FrozenSet[str] = frozenset({"sae", "targeted_replication"})

_ALIAS = {
    "CounterFact": "counterfact",
    "IOI": "ioi",
    "GreaterThan": "greater_than",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "SimpleCopying": "simple_copying",
    "MMLUPro": "mmlu_pro",
    "BBH": "bbh",
    "MedQA": "medqa",
    "MATH500": "math500",
}


def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, _TASKS, "qwen3_1_7b")
    allowed = set(behaviors_for_model("qwen3_1_7b"))
    for beh in list(allowed):
        if beh in _ALIAS:
            allowed.add(_ALIAS[beh])
    assert_prompt_source_in(cfg, frozenset(allowed), "qwen3_1_7b")
