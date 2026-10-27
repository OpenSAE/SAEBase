"""SAEBase live capture task."""
from tasks.LLM.sae_runtime import SAEOperationTask


class CaptureTask(SAEOperationTask):
    OPERATION = "capture"
