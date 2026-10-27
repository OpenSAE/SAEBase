"""Register NLP prompt dataset plugins.

Each concrete dataset owns its source and preparation logic under its family;
the root plugin table only dispatches to those implementations.
"""

from __future__ import annotations
from saebase.research_registry import register_dataset

DATASET_NAMES = ("CounterFact", "GreaterThan", "IOI", "MMLUPro", "SimpleCopying")
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
    def setup_dataset(dm, cfg, stage=None):
        from research_datasets.prompts import read_prompts

        records = read_prompts(cfg.dataset, model_key=str(cfg.model.name))
        dm.train_set = [item for item in records if str(item.get("split", "train")) == "train"]
        dm.val_set = [item for item in records if item.get("split") in ("val", "validation")]
        dm.test_set = [item for item in records if item.get("split") in ("test", "held_out")]
        if not dm.train_set and not dm.val_set and not dm.test_set:
            raise ValueError("No supported train/val/test split among NLP prompt records")

    def check_dataset(cfg):
        name, task = str(cfg.dataset.name), str(cfg.task.name)
        if name not in DATASET_NAMES or task not in SAE_TASKS:
            raise ValueError(f"NLP dataset {name!r} supports SAEBase tasks, received {task!r}")
        from research_datasets.artifacts import assert_available

        assert_available(name, str(cfg.model.name))

    def split_note(dm):
        return (
            "NLP prompt splits are defined only by explicit input record labels; "
            "no validation or test set is inferred from training prompts."
        )

    for name in DATASET_NAMES:
        register_dataset(name, setup_dataset, check_dataset, split_note)


_register_builtin_datasets()
