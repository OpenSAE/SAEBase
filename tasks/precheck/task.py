"""Reusable SAEBase Precheck task, with no imported GPU backends on registration."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.registry import TaskModule
from saebase.results import EvaluationResult


class PrecheckTask(TaskModule):
    def validate_cfg(self, cfg: DictConfig) -> None:
        from saebase.registry import get_dataset_plugin
        from tasks.precheck.operation import _budget, _selected_tasks
        from tasks.sae_runtime import load_requested_extensions

        if "sample_count" in cfg.dataset:
            from saebase.datasets.selection import validate_selection

            validate_selection(cfg.dataset)
        load_requested_extensions(cfg)
        mode = str(cfg.task.get("mode", "sae"))
        if cfg.task.get("resource_only", False) and mode != "sae":
            raise ValueError("resource_only prepares SAE resources; use precheck mode=sae")
        tasks = _selected_tasks(cfg, mode)
        if (
            not cfg.task.get("prompt")
            and cfg.dataset.get("samples") is None
            and (mode != "task" or any(name != "benchmark" for name in tasks))
        ):
            get_dataset_plugin(str(cfg.dataset.name))
        retired = {"allow_compute", "allow_gpu", "allow_native"} & set(cfg.task)
        if retired:
            raise ValueError(
                f"Precheck execution no longer requires extra permission flags: {sorted(retired)}; select device and actual check plans"
            )
        for field in (
            "prepare_timeout_seconds",
            "dataset_prepare_timeout_seconds",
            "dataset_n_generate",
            "dataset_min_kept",
            "check_sample_count",
        ):
            if field in cfg.task and (type(cfg.task[field]) is not int or cfg.task[field] < 1):
                raise ValueError(f"task.{field} must be a positive integer")
        if cfg.task.get("dataset_min_kept", 1) > cfg.task.get("dataset_n_generate", 32):
            raise ValueError("dataset_min_kept exceeds dataset_n_generate")
        for field in (
            "allow_network",
            "allow_install",
            "allow_download",
            "auto_setup",
            "resource_only",
        ):
            if field in cfg.task and type(cfg.task[field]) is not bool:
                raise ValueError(f"task.{field} must be a boolean")
        if (
            type(cfg.task.get("max_download_bytes", 16 * 1024**3)) is not int
            or cfg.task.get("max_download_bytes", 16 * 1024**3) < 0
        ):
            raise ValueError("max_download_bytes must be a nonnegative integer")
        mode = str(cfg.task.get("mode", "sae"))
        if mode not in {"sae", "task", "all"}:
            raise ValueError("Precheck mode must be sae, task or all")
        _budget(cfg)
        _selected_tasks(cfg, mode)

    @property
    def name(self) -> str:
        return "precheck"

    def run(self, cfg: DictConfig) -> EvaluationResult:
        self.validate_cfg(cfg)
        return self.run_validated(cfg)

    def run_validated(self, cfg: DictConfig) -> EvaluationResult:
        from tasks.precheck.operation import run_precheck

        return run_precheck(cfg)
