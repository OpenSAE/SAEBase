"""SAEBase train sae task."""

from saebase.runtime.sae_operations import SAEOperationTask


class TrainTask(SAEOperationTask):
    OPERATION = "train"
