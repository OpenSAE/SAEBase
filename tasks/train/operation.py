"""Concrete train task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value

from pathlib import Path
from typing import Any

from saebase.runtime.sae_operations import _task, input_path


def _train(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from tasks.train.execution.training import TrainPlan, run_training

    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Native SAE training requires task.train.approved=true")
    raw = _task(cfg, "train_plan")
    if raw is None:
        raise ValueError("Provide a complete TrainPlan in task.train.train_plan")
    resolved = OmegaConf.to_container(raw, resolve=True)
    for field in ("model_path", "dataset_path", "resume_checkpoint"):
        if resolved.get(field):
            resolved[field] = str(input_path(resolved[field]))
    plan = TrainPlan(**resolved)
    if not plan.model_path or not plan.model_revision:
        raise ValueError("Training requires an exact model snapshot identity")
    summary = run_training(
        plan,
        output=Path.cwd() / "checkpoints" / "native_sae_training",
        approved=True,
        allow_large=bool(_task(cfg, "allow_large", False)),
        timeout_seconds=int(_task(cfg, "timeout_seconds", 1800)),
    )
    from saebase.results import completion_summary

    return {
        "completion": completion_summary([{"status": "completed"}]),
        "status": "executed",
        "operation": "train",
        "model_key": str(cfg.model.name),
        "native": summary,
        "resolved_protocol": {
            "launch_labels": {
                "model": str(cfg.model.name),
                "dataset": str(getattr(getattr(cfg, "dataset", None), "name", "")),
            },
            "model_snapshot": plan.model_path,
            "model_revision": plan.model_revision,
            "dataset_snapshot": plan.dataset_path,
            "dataset_revision": plan.dataset_revision,
            "device": plan.device,
        },
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _train(cfg)


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from saebase.runtime.sae_operations import validate_fields, validate_native_selection
    from omegaconf import OmegaConf
    from tasks.train.execution.training import TrainPlan

    validate_native_selection(cfg)
    validate_fields(cfg, integers=(("timeout_seconds", 1),), booleans=("approved", "allow_large"))
    raw = task_value(cfg.task, "train_plan")
    if raw is None:
        if cfg.task.get("execute", False):
            raise ValueError("Training requires task.train.train_plan")
        return
    raw = OmegaConf.to_container(raw, resolve=True)
    for field in ("model_path", "dataset_path", "resume_checkpoint"):
        if raw.get(field):
            raw[field] = str(input_path(raw[field]))
    TrainPlan(**raw).validate(allow_large=bool(task_value(cfg.task, "allow_large", False)))


def prepare_execution(cfg: Any) -> dict:
    """Prepare the selected protocol without running the research calculation."""
    from importlib.util import find_spec

    backend = task_value(cfg.task, "train_plan").backend
    module = "sae_lens" if backend == "saelens" else "sparsify"
    if find_spec(module) is None:
        raise ImportError(f"Install the selected native training backend: {module}")
    return {
        "status": "PASS",
        "scope": "native training plan, local inputs and backend ready; no training executed",
        "steps": task_value(cfg.task, "train_plan").steps,
    }
