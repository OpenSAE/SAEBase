"""Reusable SAEBase Precheck task, with no imported GPU backends on registration."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.registry import TaskModule
from saebase.results import EvaluationResult


class PrecheckTask(TaskModule):
    @property
    def name(self) -> str:
        return "precheck"

    def run(self, cfg: DictConfig) -> EvaluationResult:
        from tasks.precheck.operation import run_precheck

        return run_precheck(cfg)
