"""Reusable SAEBase Precheck task, with no imported GPU backends on registration."""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_owner, task_value

from omegaconf import DictConfig, OmegaConf

from saebase.registry import TaskModule
from saebase.results import EvaluationResult


class PrecheckTask(TaskModule):
    def validate_cfg(self, cfg: DictConfig) -> None:
        from saebase.registry import get_dataset_plugin
        from tasks.precheck.operation import _budget, _selected_tasks
        from saebase.runtime.sae_operations import load_requested_extensions

        if "sample_count" in cfg.dataset:
            from saebase.datasets.selection import validate_selection

            validate_selection(cfg.dataset)
        load_requested_extensions(cfg)
        mode = str(task_value(cfg.task, "mode", "sae"))
        if mode == "performance":
            from saebase.runtime.performance import settings

            settings(cfg)
            if any(
                name not in {"capture", "intervene", "generate"}
                for name in task_value(cfg.task, "check_tasks", [])
            ):
                raise ValueError("Performance Precheck supports capture, intervene and generate")
            targets = list(task_value(cfg.task, "check_tasks", []))
            if len(set(targets)) != len(targets):
                raise ValueError("Performance check_tasks must be unique")
            selection = task_value(cfg.task, "sae_list", "default")
            if selection == "all" or (OmegaConf.is_list(selection) and len(selection) != 1):
                raise ValueError(
                    "Performance Precheck selects one SAE/site; run different resources separately"
                )
            modes = list(task_value(cfg.task, "capture_modes", ["whiten", "non_whiten", "g_s", "I"]))
            if (
                not modes
                or len(set(modes)) != len(modes)
                or set(modes) - {"whiten", "non_whiten", "g_s", "I", "sar"}
            ):
                raise ValueError("capture_modes must contain distinct registered Capture modes")
            return
        if task_value(cfg.task, "resource_only", False) and mode != "sae":
            raise ValueError("resource_only prepares SAE resources; use precheck mode=sae")
        tasks = _selected_tasks(cfg, mode)
        if (
            not cfg.task.get("prompt")
            and cfg.dataset.get("samples") is None
            and (mode != "task" or any(name != "benchmark" for name in tasks))
        ):
            get_dataset_plugin(str(cfg.dataset.name))
        retired = {"allow_compute", "allow_gpu", "allow_native"} & (set(cfg.task) | set(task_owner(cfg.task)))
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
            value = task_value(cfg.task, field, 1)
            if type(value) is not int or value < 1:
                raise ValueError(f"task.precheck.{field} must be a positive integer")
        if task_value(cfg.task, "dataset_min_kept", 1) > task_value(cfg.task, "dataset_n_generate", 32):
            raise ValueError("dataset_min_kept exceeds dataset_n_generate")
        for field in (
            "allow_network",
            "allow_install",
            "allow_download",
            "auto_setup",
            "resource_only",
        ):
            if type(task_value(cfg.task, field, False)) is not bool:
                raise ValueError(f"task.precheck.{field} must be a boolean")
        if (
            type(task_value(cfg.task, "max_download_bytes", 16 * 1024**3)) is not int
            or task_value(cfg.task, "max_download_bytes", 16 * 1024**3) < 0
        ):
            raise ValueError("max_download_bytes must be a nonnegative integer")
        mode = str(task_value(cfg.task, "mode", "sae"))
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
        if task_value(cfg.task, "mode") == "performance":
            from tasks.precheck.performance import run_performance_precheck

            return run_performance_precheck(cfg)
        from tasks.precheck.operation import run_precheck

        return run_precheck(cfg)
