"""Validate supported Gemma-2 2B task and NLP dataset combinations."""
from __future__ import annotations
from omegaconf import DictConfig
from datasets.NLP.artifacts import assert_available

SAE_TASKS = frozenset({"capture", "intervene", "generate",
                       "circuits", "explain", "benchmark", "train"})
DATASETS = {"CounterFact", "GreaterThan", "IOI", "MMLUPro", "SimpleCopying"}

def assert_compatible(cfg: DictConfig) -> None:
    if str(cfg.model.name) != "gemma2_2b_base":
        raise ValueError("Gemma-2 2B model identity required")
    if str(cfg.task.name) not in SAE_TASKS:
        raise ValueError(f"Gemma-2 2B only supports SAE tasks: {sorted(SAE_TASKS)}")
    name = str(cfg.dataset.name)
    if name not in DATASETS:
        raise ValueError(f"Unsupported Gemma-2 2B prompt dataset: {name}")
    assert_available(name, "gemma2_2b_base")
