"""SAEBase generative intervention task."""

from tasks.sae_runtime import SAEOperationTask


class GenerateTask(SAEOperationTask):
    OPERATION = "generate"
