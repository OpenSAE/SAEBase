"""SAEBase teacher forced intervention task."""

from saebase.runtime.configuration.task_options import task_value

from saebase.runtime.sae_operations import SAEOperationTask


class InterveneTask(SAEOperationTask):
    OPERATION = "intervene"

    def supports_run_resume(self, cfg):
        return task_value(cfg.task, "protocol") is not None or task_value(cfg.task, "feature_selection") == "capture_top"
