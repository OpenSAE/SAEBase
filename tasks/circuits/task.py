"""SAEBase circuits task."""

from tasks.sae_runtime import SAEOperationTask


class CircuitsTask(SAEOperationTask):
    OPERATION = "circuits"
