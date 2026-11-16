"""SAEBase teacher forced intervention task."""

from tasks.sae_runtime import SAEOperationTask


class InterveneTask(SAEOperationTask):
    OPERATION = "intervene"

    def supports_run_resume(self, cfg):
        return cfg.task.get("protocol") is not None or cfg.task.get("feature_selection") == "capture_top"
