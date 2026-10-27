"""SAEBase teacher forced intervention task."""

from tasks.sae_runtime import SAEOperationTask


class InterveneTask(SAEOperationTask):
    OPERATION = "intervene"
