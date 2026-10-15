"""task × prompt-source compatibility for Qwen3.5 2B."""
from __future__ import annotations

from omegaconf import DictConfig

from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in
from saebase.data.prompt_schema import behaviors_for_model

_ALIAS = {
    "CounterFact": "counterfact", "IOI": "ioi",
    "GreaterThan": "greater_than",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "SimpleCopying": "simple_copying",
    "MMLUPro": "mmlu_pro",
}


def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, frozenset({"sae"}), "qwen35_2b")
    allowed = set(behaviors_for_model("qwen35_2b"))
    allowed.update(_ALIAS[k] for k in allowed if k in _ALIAS)
    assert_prompt_source_in(cfg, frozenset(allowed), "qwen35_2b")
