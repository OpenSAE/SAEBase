"""Concrete train task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tasks.sae_runtime import _task


def _train(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from tasks.train.backend.training import TrainPlan, run_training

    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Native SAE training requires task.approved=true")
    raw = _task(cfg, "train_plan")
    if raw is None:
        raise ValueError("Provide a complete TrainPlan in task.train_plan")
    plan = TrainPlan(**OmegaConf.to_container(raw, resolve=True))
    if not plan.model_path or not plan.model_revision:
        raise ValueError("Training requires an exact model snapshot identity")
    summary = run_training(
        plan,
        output=Path.cwd() / "checkpoints" / "native_sae_training",
        approved=True,
        allow_large=bool(_task(cfg, "allow_large", False)),
        timeout_seconds=int(_task(cfg, "timeout_seconds", 1800)),
    )
    return {"status": "executed", "operation": "train", "model_key": str(cfg.model.name), "native": summary}


def execute(cfg: Any) -> dict[str, Any]:
    return _train(cfg)
