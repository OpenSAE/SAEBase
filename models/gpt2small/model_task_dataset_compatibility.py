"""Standard registered SAE tasks and text behaviors for GPT-2 Small."""
from __future__ import annotations
from omegaconf import DictConfig
from saebase.cfg.contracts import assert_prompt_source_in,assert_task_in
from saebase.datasets.prompt_schema import behaviors_for_model

_ALIASES={"CounterFact":"counterfact","SimpleCopying":"simple_copying",
          "SubjectVerbAgreement":"subject_verb_agreement"}

def assert_compatible(cfg:DictConfig)->None:
    assert_task_in(cfg,frozenset({"sae"}),"gpt2small")
    allowed=set(behaviors_for_model("gpt2small"))
    allowed.update(_ALIASES[x] for x in allowed if x in _ALIASES)
    assert_prompt_source_in(cfg,frozenset(allowed),"gpt2small")
