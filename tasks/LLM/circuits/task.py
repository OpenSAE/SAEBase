"""SAEBase circuits task."""
from tasks.LLM.sae_runtime import SAEOperationTask


class CircuitsTask(SAEOperationTask):
    OPERATION = "circuits"
