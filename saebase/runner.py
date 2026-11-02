"""Dispatch ``cfg.task.name`` to a concrete task package."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.results import EvaluationResult


def run_task(cfg: DictConfig) -> EvaluationResult:
    from saebase.registry import get_task_plugin

    if str(cfg.task.name) == "sae":
        # Legacy standalone CLI keeps its activation-health entrypoint without
        # occupying one of the eight current Task Registry keys.
        from saebase.sae.task import SAETask
        return SAETask().run(cfg)
    return get_task_plugin(str(cfg.task.name)).task_class().run(cfg)
