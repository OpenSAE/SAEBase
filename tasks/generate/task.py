"""SAEBase generative intervention task."""

from saebase.runtime.configuration.task_options import task_value

from saebase.runtime.sae_operations import SAEOperationTask


class GenerateTask(SAEOperationTask):
    OPERATION = "generate"

    def supports_run_resume(self, cfg):
        return task_value(cfg.task, "feature_selection") == "capture_top"
