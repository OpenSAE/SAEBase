"""Dispatch a configured research task and require the shared task result contract."""

from __future__ import annotations
from omegaconf import DictConfig, OmegaConf
from saebase.results import EvaluationResult
from saebase.registry import TaskModule


def run_task(cfg: DictConfig, *, task: TaskModule | None = None) -> EvaluationResult:
    from models.loading_policy import resolve_config_devices

    from saebase.registry import get_task_plugin

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolve_config_devices(cfg)

    if task is None:
        task = get_task_plugin(str(cfg.task.name)).task_class()
        task.validate_cfg(cfg)
    receipt = None
    if str(cfg.task.name) != "precheck" and bool(cfg.task.get("execute", False)):
        from tasks.precheck.gate import automatic_precheck

        blocked, receipt = automatic_precheck(cfg)
        if blocked is not None:
            return blocked
    result = getattr(task, "run_validated", task.run)(cfg)
    if not isinstance(result, EvaluationResult):
        raise TypeError(f"Task {cfg.task.name!r} must return EvaluationResult")
    if receipt is not None:
        result.artifact_paths["precheck"] = receipt
    return result
