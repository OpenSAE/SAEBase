"""SAE task and prompt-source compatibility for mistral7b_instruct."""
from __future__ import annotations
from omegaconf import DictConfig
from saebase.data.prompt_schema import behaviors_for_model
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in

_ALIASES = {"CounterFact":"counterfact","IOI":"ioi","GreaterThan":"greater_than",
            "SubjectVerbAgreement":"subject_verb_agreement",
            "SimpleCopying":"simple_copying","MMLUPro":"mmlu_pro"}

def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg,frozenset({"sae"}),"mistral7b_instruct")
    allowed=set(behaviors_for_model("mistral7b_instruct"))
    allowed.update(_ALIASES[x] for x in list(allowed) if x in _ALIASES)
    assert_prompt_source_in(cfg,frozenset(allowed),"mistral7b_instruct")
