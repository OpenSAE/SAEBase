"""SAEBase generative intervention task."""

from tasks.LLM.sae_runtime import SAEOperationTask


class GenerateTask(SAEOperationTask):
    OPERATION = "generate"
