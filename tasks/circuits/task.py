"""SAEBase circuits task."""

from saebase.runtime.sae_operations import SAEOperationTask


class CircuitsTask(SAEOperationTask):
    OPERATION = "circuits"
    precheck_prepares_only = True
