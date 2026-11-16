"""SAEBase generative intervention task."""

from tasks.sae_runtime import SAEOperationTask


class GenerateTask(SAEOperationTask):
    OPERATION = "generate"

    def supports_run_resume(self, cfg):
        return cfg.task.get("feature_selection") == "capture_top"
