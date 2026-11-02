"""task × prompt_source compatibility for ``gemma2_2b_base``."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.datasets.prompt_schema import behaviors_for_model
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
    if str(cfg.task.name) != "sae":
        from saebase.datasets.prompt_schema import assert_available

        task = str(cfg.task.name)
        if task not in {
            "capture",
            "intervene",
            "generate",
            "circuits",
            "explain",
            "benchmark",
            "train",
        }:
            raise ValueError(f"Unsupported SAE task: {task}")
        name = str(cfg.dataset.name)
        assert_available(name, "gemma2_2b_base")
        return
    assert_task_in(cfg, _TASKS, "gemma2_2b_base")
    allowed = set(behaviors_for_model("gemma2_2b_base"))
    for beh in list(allowed):
        if beh in _ALIAS:
            allowed.add(_ALIAS[beh])
    assert_prompt_source_in(cfg, frozenset(allowed), "gemma2_2b_base")
