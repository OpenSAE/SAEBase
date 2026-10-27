"""SAEBase train sae task."""

from tasks.sae_runtime import SAEOperationTask


class TrainTask(SAEOperationTask):
    OPERATION = "train"
