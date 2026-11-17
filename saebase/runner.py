"""Dispatch a configured research task and require the shared task result contract."""

from __future__ import annotations
from contextlib import ExitStack
from omegaconf import DictConfig, OmegaConf
from saebase.results import EvaluationResult
from saebase.registry import TaskModule


def run_task(cfg: DictConfig, *, task: TaskModule | None = None) -> EvaluationResult:
    from models.loading_policy import resolve_config_devices

    from saebase.registry import get_task_plugin

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolve_config_devices(cfg)

    from models.numerics import configure_task
    from saebase.runtime.environment import configure_worker

    configure_worker(cfg)
    configure_task(cfg)

    if task is None:
        task = get_task_plugin(str(cfg.task.name)).task_class()
        task.validate_cfg(cfg)
    receipt = None
    from saebase.runtime.performance import session, stage, settings, consume_recommendations

    with session(cfg) as performance, ExitStack() as resources:
        if (
            settings(cfg)["enabled"]
            and str(cfg.task.name) in {"capture", "intervene", "generate"}
            and cfg.task.get("execute", False)
        ):
            from tasks.sae_runtime import _pair, loaded_pair_session

            with stage("model_and_sae_load"):
                pair = _pair(cfg)
            resources.enter_context(loaded_pair_session(pair))
        blocked = None
        if str(cfg.task.name) != "precheck" and bool(cfg.task.get("execute", False)):
            from tasks.precheck.gate import automatic_precheck

            with stage("automatic_precheck"):
                blocked, receipt = automatic_precheck(cfg)
            consume_recommendations(performance)
        if blocked is not None:
            result = blocked
        else:
            with stage("task_execution"):
                result = getattr(task, "run_validated", task.run)(cfg)
    if not isinstance(result, EvaluationResult):
        raise TypeError(f"Task {cfg.task.name!r} must return EvaluationResult")
    if cfg.task.get("execute", False):
        from models.numerics import identity as numerical_identity

        result.meta["numerics"] = numerical_identity()
        from saebase.runtime.environment import worker_snapshot

        result.meta["worker"] = worker_snapshot()
    if receipt is not None:
        result.artifact_paths["precheck"] = receipt
    if performance is not None:
        from pathlib import Path

        result.artifact_paths.setdefault(
            "performance", str(Path.cwd() / "reports/performance.json")
        )
        result.artifact_paths.setdefault(
            "performance_report", str(Path.cwd() / "reports/performance.md")
        )
    return result
