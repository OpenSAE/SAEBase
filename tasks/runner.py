"""Dispatch ``cfg.task.name`` to a concrete task package."""

from __future__ import annotations

from omegaconf import DictConfig


def run_task(cfg: DictConfig):
    from dlbase.registry import get_task_plugin

    return get_task_plugin(str(cfg.task.name)).task_class().run(cfg)
