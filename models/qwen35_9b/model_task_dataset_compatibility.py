"""task × prompt_source compatibility for ``qwen35_9b``."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.data.prompt_schema import behaviors_for_model
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in

_TASKS: FrozenSet[str] = frozenset({"sae"})

_ALIAS = {
    "CounterFact": "counterfact",
    "IOI": "ioi",
    "GreaterThan": "greater_than",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "SimpleCopying": "simple_copying",
    "MMLUPro": "mmlu_pro",
}


def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, _TASKS, "qwen35_9b")
    allowed = set(behaviors_for_model("qwen35_9b"))
    for beh in list(allowed):
        if beh in _ALIAS:
            allowed.add(_ALIAS[beh])
    assert_prompt_source_in(cfg, frozenset(allowed), "qwen35_9b")
