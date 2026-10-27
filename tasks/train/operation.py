"""Concrete train task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tasks.sae_runtime import MODEL_KEY, _task


def _train(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from tasks.train.backend.training import TrainPlan, run_training

    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Native SAE training requires task.approved=true")
    raw = _task(cfg, "train_plan")
    if raw is None:
        raise ValueError("Provide a complete TrainPlan in task.train_plan")
    plan = TrainPlan(**OmegaConf.to_container(raw, resolve=True))
    if MODEL_KEY not in plan.model_path and "gemma" not in plan.model_path.lower():
        raise ValueError("Training model snapshot path must identify the selected Gemma-2 2B model")
    summary = run_training(
        plan,
        output=Path.cwd() / "checkpoints" / "native_sae_training",
        approved=True,
        allow_large=bool(_task(cfg, "allow_large", False)),
        timeout_seconds=int(_task(cfg, "timeout_seconds", 1800)),
    )
    return {"status": "executed", "operation": "train", "model_key": MODEL_KEY, "native": summary}


def execute(cfg: Any) -> dict[str, Any]:
    return _train(cfg)
