"""SAEBase automatic feature-explanation task, requiring a real generator."""

from __future__ import annotations
from saebase.results import EvaluationResult
from saebase.runtime.sae_operations import SAEOperationTask, execute_task


class ExplainTask(SAEOperationTask):
    OPERATION = "explain"

    def __init__(self, generator=None, predictor=None):
        self.generator = generator
        self.predictor = predictor

    def run_validated(self, cfg) -> EvaluationResult:
        payload = execute_task(
            self.OPERATION, cfg, generator=self.generator, predictor=self.predictor, validated=True
        )
        return EvaluationResult.from_payload(self.OPERATION, cfg, payload)
