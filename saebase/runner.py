"""Dispatch ``cfg.task.name`` to a concrete task package."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.results import EvaluationResult


def run_task(cfg: DictConfig) -> EvaluationResult:
    from saebase.registry import get_task_plugin

    return get_task_plugin(str(cfg.task.name)).task_class().run(cfg)
