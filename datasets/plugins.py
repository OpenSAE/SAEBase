"""Register prompt-source plugins (imported by ``saebase.plugin_bootstrap``)."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.datasets.artifact_layout import BEHAVIOR_ALIASES, canonicalize_behavior
from saebase.datasets.prompt_schema import assert_available
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in
from saebase.registry import (
    register_prompt_source,
    register_prompt_source_alias,
    register_dataset,
    register_dataset_alias,
)

_TASKS_SAE: FrozenSet[str] = frozenset({"sae"})
_TASKS_DISCOVERY: FrozenSet[str] = frozenset({"sae", "discovery", "logic_chain"})
_TASKS_STEER: FrozenSet[str] = frozenset(
    {"sae", "discovery", "logic_chain", "vllm_sae_steer", "pair_interaction", "ordered_steering"}
)

_BEHAVIORS = (
    ("CounterFact", "counterfact"),
    ("IOI", "ioi"),
    ("GreaterThan", "greater_than"),
    ("SubjectVerbAgreement", "subject_verb_agreement"),
    ("SimpleCopying", "simple_copying"),
    ("MMLUPro", "mmlu_pro"),
    ("BBH", "bbh"),
    ("ARCChallenge", "arc_challenge"),
    ("MedQA", "medqa"),
    ("MATH500", "math500"),
    ("GPQAExtended", "gpqa_extended"),
    ("LiveCodeBench", "livecodebench"),
)


def _tasks_for(canonical: str) -> FrozenSet[str]:
    if canonical in {"MMLUPro", "BBH"} | {
        "ARCChallenge",
        "MedQA",
        "MATH500",
        "GPQAExtended",
        "LiveCodeBench",
    }:
        return _TASKS_STEER
    return _TASKS_SAE


def _make_assert(canonical: str):
    alias_set = frozenset(k for k, v in BEHAVIOR_ALIASES.items() if v == canonical) | frozenset(
        {canonical}
    )
    allowed_tasks = _tasks_for(canonical)

    def assert_compatible(cfg: DictConfig) -> None:
        assert_task_in(cfg, allowed_tasks, canonical)
        assert_prompt_source_in(cfg, alias_set, canonical)
        assert_available(canonical, str(cfg.model.name))

    return assert_compatible


def _notes(canonical: str):
    def notes() -> str:
        return (
            f"Prompt source {canonical}: prepared artifacts under data_dir/{canonical}/<model>/v1"
        )

    return notes


def _register_builtin_prompt_sources() -> None:
    for canonical, alias in _BEHAVIORS:
        register_prompt_source(
            canonical,
            _make_assert(canonical),
            canonicalize_behavior,
            _notes(canonical),
        )
        register_prompt_source_alias(alias, canonical)


_register_builtin_prompt_sources()


NLP_NAMES = ("CounterFact", "GreaterThan", "IOI", "MMLUPro", "SimpleCopying")
SAE_TASKS = frozenset(
    {
        "capture",
        "intervene",
        "generate",
        "circuits",
        "explain",
        "benchmark",
        "train",
    }
)


def _register_builtin_datasets() -> None:
    def setup_nlp(dm, cfg, stage=None):
        from saebase.datasets.prompts import read_prompts

        records = read_prompts(cfg.dataset, model_key=str(cfg.model.name))
        dm.train_set = [item for item in records if str(item.get("split", "train")) == "train"]
        dm.val_set = [item for item in records if item.get("split") in ("val", "validation")]
        dm.test_set = [item for item in records if item.get("split") in ("test", "held_out")]
        if not dm.train_set and not dm.val_set and not dm.test_set:
            raise ValueError("No supported train/val/test split among NLP prompt records")

    def check_nlp(cfg):
        name, task = str(cfg.dataset.name), str(cfg.task.name)
        if name not in NLP_NAMES or task not in SAE_TASKS:
            raise ValueError(f"NLP dataset {name!r} supports SAEBase tasks, received {task!r}")
        from saebase.datasets.artifacts import assert_available

        assert_available(name, str(cfg.model.name))

    def nlp_split_note(dm):
        return (
            "NLP prompt splits are defined only by explicit input record labels; "
            "no validation or test set is inferred from training prompts."
        )

    for name in NLP_NAMES:
        adapter = "saebase.datasets.MMLUPro.qa_adapter" if name == "MMLUPro" else None
        register_dataset(
            name,
            setup_nlp,
            check_nlp,
            nlp_split_note,
            qa_adapter_module=adapter,
        )


_register_builtin_datasets()


def _register_qa_dataset_adapters() -> None:
    """Register source-owned QA adapters on the existing Dataset registry.

    This augments reusable dataset capabilities; experiment selection belongs
    exclusively to an experiment config, not to this module.
    """
    adapters = {
        "ARCChallenge": ("arc_challenge", "ARCChallenge"),
        "CommonsenseQA": ("commonsense_qa", "CommonsenseQA"),
        "BoolQ": ("boolq", "BoolQ"),
        "BBH": ("bbh", "BBH"),
        "GSM8K": ("gsm8k", "GSM8K"),
        "MATH500": ("math500", "MATH500"),
        "MedQA": ("medqa", "MedQA"),
        "HotpotQA": ("hotpot_qa", "HotpotQA"),
        "GPQAExtended": ("gpqa_extended", "GPQAExtended"),
    }

    def qa_setup(dm, cfg, stage=None):
        from saebase.registry import get_qa_dataset_adapter
        from saebase.data.qa_records import make_record
        from saebase.data.qa_source import iter_source

        adapter = get_qa_dataset_adapter(str(cfg.dataset.name))
        revision = str(cfg.dataset.get("source_revision", "")).strip()
        if len(revision) != 40:
            raise ValueError("QA Dataset requires an explicitly pinned 40-char revision")
        data = []
        for config, index, row in iter_source(adapter.SPEC, revision):
            parsed = adapter.parse(row, source_config=config, seed=int(cfg.get("seed", 0)))
            data.append(make_record(adapter.SPEC, row, parsed, config, index))
        dm.train_set = data if adapter.SPEC.split == "train" else []
        dm.val_set = data if adapter.SPEC.split == "validation" else []
        dm.test_set = data if adapter.SPEC.split == "test" else []

    def qa_check(cfg):
        if str(cfg.task.name) not in SAE_TASKS:
            raise ValueError("QA dataset must be paired with a registered SAEBase task")

    def qa_note(dm):
        return "Upstream QA source split is pinned; no extra split is inferred"

    for name, (alias, directory) in adapters.items():
        register_dataset(
            name,
            qa_setup,
            qa_check,
            qa_note,
            qa_adapter_module=f"saebase.datasets.{directory}.qa_adapter",
        )
        register_dataset_alias(alias, name)
    register_dataset_alias("mmlu_pro", "MMLUPro")


_register_qa_dataset_adapters()
