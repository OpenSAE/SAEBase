"""Task and prompt compatibility for Qwen3.5-2B Base."""
from __future__ import annotations

from typing import FrozenSet
from omegaconf import DictConfig

from saebase.datasets.prompt_schema import behaviors_for_model
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in

_TASKS: FrozenSet[str] = frozenset({"sae"})
_ALIASES = {
    "CounterFact": "counterfact",
    "IOI": "ioi",
    "GreaterThan": "greater_than",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "SimpleCopying": "simple_copying",
    "MMLUPro": "mmlu_pro",
}

def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, _TASKS, "qwen35_2b")
    allowed = set(behaviors_for_model("qwen35_2b"))
    allowed |= {_ALIASES[k] for k in allowed if k in _ALIASES}
    assert_prompt_source_in(cfg, frozenset(allowed), "qwen35_2b")
