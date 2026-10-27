"""SAEBase automatic feature-explanation task, requiring a real generator."""

from __future__ import annotations
from tasks.LLM.sae_runtime import SAEOperationTask, execute_task


class ExplainTask(SAEOperationTask):
    OPERATION = "explain"

    def __init__(self, generator=None, predictor=None):
        self.generator = generator
        self.predictor = predictor

    def run(self, cfg):
        return execute_task(self.OPERATION, cfg, generator=self.generator, predictor=self.predictor)
