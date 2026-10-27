"""Register shared Vision and NLP dataset plugins.

Each concrete dataset owns its source and preparation logic under its family;
the root plugin table only dispatches to those implementations.
"""
from __future__ import annotations
from dlbase.registry import register_dataset, register_dataset_alias

NLP_NAMES = ("CounterFact", "GreaterThan", "IOI", "MMLUPro", "SimpleCopying")
SAE_TASKS = frozenset({
    "capture", "intervene", "generate",
    "circuits", "explain", "benchmark", "train",
})

def _register_builtin_datasets() -> None:
    def setup_cub_datamodule(dm, cfg, stage=None):
        from research_datasets.vision.CUB_200.datamodule_setup import setup_cub_datamodule
        return setup_cub_datamodule(dm, cfg, stage)

    def cub_assert(cfg):
        from research_datasets.vision.CUB_200.dataset_task_policy import assert_task_supported
        return assert_task_supported(cfg)

    def cub_eval_split_note(dm):
        from research_datasets.vision.CUB_200.datamodule_setup import cub_eval_split_note
        return cub_eval_split_note(dm)

    register_dataset("CUB_200", setup_cub_datamodule, cub_assert, cub_eval_split_note)
    register_dataset_alias("CUB", "CUB_200")

    def setup_nlp(dm, cfg, stage=None):
        from research_datasets.NLP.prompts import read_prompts
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
        from research_datasets.NLP.artifacts import assert_available
        assert_available(name, str(cfg.model.name))

    def nlp_split_note(dm):
        return ("NLP prompt splits are defined only by explicit input record labels; "
                "no validation or test set is inferred from training prompts.")

    for name in NLP_NAMES:
        register_dataset(name, setup_nlp, check_nlp, nlp_split_note)

_register_builtin_datasets()
