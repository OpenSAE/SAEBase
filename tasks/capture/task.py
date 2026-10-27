"""SAEBase live capture task."""

from tasks.sae_runtime import SAEOperationTask


class CaptureTask(SAEOperationTask):
    OPERATION = "capture"
