"""SAEBase teacher forced intervention task."""
from tasks.LLM.sae_runtime import SAEOperationTask


class InterveneTask(SAEOperationTask):
    OPERATION = "intervene"
