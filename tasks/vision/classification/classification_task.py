"""Classification task: Lightning training for CUB / I-RAVEN."""

from __future__ import annotations

from omegaconf import DictConfig

from dlbase.registry import TaskModule


class ClassificationTask(TaskModule):
    @property
    def name(self) -> str:
        return "classification"

    def run(self, cfg: DictConfig) -> None:
        from tasks.vision.classification.run import run_classification_training
        run_classification_training(cfg)
