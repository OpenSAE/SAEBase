"""Register prompt-source plugins (imported by ``saebase.plugin_bootstrap``)."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.datasets.artifacts import BEHAVIOR_ALIASES, canonicalize_behavior
from saebase.datasets.artifacts import assert_available
from saebase.runtime.cfg.contracts import assert_prompt_source_in, assert_task_in
from saebase.registry import register_prompt_source, register_prompt_source_alias, register_dataset

_TASKS_SAE: FrozenSet[str] = frozenset({"benchmark"})
_TASKS_DISCOVERY: FrozenSet[str] = frozenset(
    {"benchmark", "capture", "intervene", "generate", "circuits", "explain", "train"}
)
_TASKS_STEER: FrozenSet[str] = frozenset(
    {"benchmark", "capture", "intervene", "generate", "circuits", "explain", "train"}
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


NLP_NAMES = (
    "HotpotQA",
    "BBH",
    "CounterFact",
    "GreaterThan",
    "IOI",
    "MMLUPro",
    "SimpleCopying",
    "RAVEL",
    "ARCChallenge",
    "GSM8K",
    "HumanEval",
    "TruthfulQA",
    "HarmBench",
    "XSTest",
)
SAE_TASKS = frozenset(
    {
        "precheck",
        "capture",
        "intervene",
        "generate",
        "circuits",
        "explain",
        "benchmark",
        "train",
    }
)


PUBLIC_DATASETS = frozenset(
    [
        "BBH",
        "RAVEL",
        "ARCChallenge",
        "GSM8K",
        "HumanEval",
        "TruthfulQA",
        "HarmBench",
        "XSTest",
        "HotpotQA",
    ]
)


def _dataset_module(name):
    import importlib

    suffix = "" if name in PUBLIC_DATASETS else ".prepare"
    return importlib.import_module(f"saebase.datasets.{name}{suffix}")


def _register_builtin_datasets() -> None:
    def preparation(name):
        def prepare(model_key, model, **parameters):
            import inspect

            operation = _dataset_module(name).prepare_for_model
            if "n_generate" not in inspect.signature(operation).parameters:
                parameters["max_items"] = parameters.pop("n_generate")
            return operation(model_key, model, **parameters)

        return prepare

    def setup_nlp(dm, cfg, stage=None):
        from saebase.datasets.prompts import read_prompts

        records = read_prompts(cfg.dataset, model_key=str(cfg.model.name))
        dm.observation_set = records
        if "sample_count" in cfg.dataset and not cfg.dataset.get("use_original_split", False):
            dm.train_set = records if str(cfg.task.name) == "train" else []
            dm.val_set = []
            dm.test_set = records if str(cfg.task.name) != "train" else []
            return
        dm.train_set = [item for item in records if str(item.get("split", "train")) == "train"]
        dm.val_set = [item for item in records if item.get("split") in ("val", "validation")]
        dm.test_set = [item for item in records if item.get("split") in ("test", "held_out")]
        if not dm.train_set and not dm.val_set and not dm.test_set:
            raise ValueError("No supported train/val/test split among NLP prompt records")

    def check_nlp(cfg):
        name, task = str(cfg.dataset.name), str(cfg.task.name)
        if name in {
            "HotpotQA",
            "BBH",
            "RAVEL",
            "ARCChallenge",
            "GSM8K",
            "HumanEval",
            "TruthfulQA",
            "HarmBench",
            "XSTest",
        }:
            policy = _dataset_module(name)
            if task not in policy.SUPPORTED_TASKS:
                raise ValueError(f"Dataset {name} has no native protocol for task {task}")
        if name not in NLP_NAMES or task not in SAE_TASKS:
            raise ValueError(f"NLP dataset {name!r} supports SAEBase tasks, received {task!r}")
        from saebase.datasets.artifacts import assert_available

        assert_available(name, str(cfg.model.name))

    def nlp_split_note(dm):
        return (
            "NLP prompt splits are defined only by explicit input record labels; "
            "no validation or test set is inferred from training prompts."
        )

    def selection(name):
        def select(cfg, **parameters):

            module = _dataset_module(name)
            return module.select_samples(cfg, **parameters)

        return select

    for name in NLP_NAMES:
        register_dataset(
            name,
            setup_nlp,
            check_nlp,
            nlp_split_note,
            prepare=preparation(name),
            select_samples=selection(name)
            if name not in {"GreaterThan", "SimpleCopying"}
            else None,
        )


_register_builtin_datasets()
