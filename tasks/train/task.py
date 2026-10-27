"""SAEBase train sae task."""

from tasks.LLM.sae_runtime import SAEOperationTask


class TrainTask(SAEOperationTask):
    OPERATION = "train"
