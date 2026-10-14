"""Hydra task ``sae``: dispatch to evaluation plugins."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.registry import TaskModule
from saebase.results import EvaluationResult


class SAETask(TaskModule):
    @property
    def name(self) -> str:
        return "sae"

    def run(self, cfg: DictConfig) -> EvaluationResult:
        evaluation = str(cfg.task.get("evaluation", "activation_health"))
        if evaluation == "activation_health":
            from saebase.sae.activation_health import run_activation_health

            return run_activation_health(cfg)
        raise ValueError(
            f"Unknown task.evaluation={evaluation!r}. "
            "Supported: ['activation_health']"
        )
