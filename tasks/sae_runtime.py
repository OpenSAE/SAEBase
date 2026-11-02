"""Seven real SAEBase Task entrypoints using configured Model/SAE resources.

No model checkpoints are downloaded until task.execute=true. Every returned
measurement originates in upstream SAEBase code; unavailable resources raise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from saebase.registry import TaskModule

MODEL_KEY = "gemma2_2b_base"
SAE_KEY = "gemma2_2b_pt_l25_canonical"
TASK_NAMES = (
    "capture",
    "intervene",
    "generate",
    "circuits",
    "explain",
    "benchmark",
    "train",
)


def _task(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.task.get(key, default)


def _require_context(cfg: Any, operation: str) -> None:
    from saebase.registry import get_dataset_plugin, get_model_plugin

    if operation not in TASK_NAMES or str(cfg.task.name) != operation:
        raise ValueError("Task name must match its registered operation")
    data = get_dataset_plugin(str(cfg.dataset.name))
    if operation in {"capture", "intervene", "generate", "circuits"} and not str(
        _task(cfg, "prompt", "") or ""
    ).strip():
        data.assert_task_supported(cfg)
    get_model_plugin(str(cfg.model.name)).assert_compatible(cfg)


def _pair(cfg: Any):
    from saebase.api import load_registered
    from saebase.api.sae_profiles import get_profile

    model_key = str(cfg.model.name)
    key = str(_task(cfg, "sae_key", "") or "")
    if not key:
        defaults = list(cfg.model.get("default_saes", []))
        if not defaults:
            raise ValueError(f"Model {model_key} has no default SAE configured")
        key = str(defaults[0])
    profile = get_profile(key)
    value = _task(cfg, "layer")
    layer = int(value) if value is not None else int(profile.layers[-1])
    if model_key not in profile.allowed_model_keys() or layer not in profile.layers:
        raise ValueError(
            f"Selected SAE {key}/{layer} is incompatible with model {model_key}"
        )
    return load_registered(
        sae_key=key, layer=layer,
        device=str(_task(cfg, "device", "cpu")), model_key=model_key,
    )


def _prompt(cfg: Any) -> str:
    value = str(_task(cfg, "prompt", "") or "").strip()
    if not value:
        from saebase.datasets.prompts import read_first_prompt

        value = read_first_prompt(cfg.dataset, model_key=str(cfg.model.name))
    if not 1 <= len(value) <= 4000:
        raise ValueError("Use a nonempty prompt up to 4000 characters")
    return value


def _fingerprint_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _maybe_save(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") == "configured_only":
        return payload
    outdir = Path.cwd() / "metadata"
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{operation}_result.json"
    if dest.exists():
        raise FileExistsError("A prior task result already exists: " + str(dest))
    blob = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    dest.write_text(blob + "\n", encoding="utf-8")
    return {**payload, "result_path": str(dest)}


# Each task owns its concrete execution implementation; SAEBase stays generic.
RUNNER_MODULES = {
    "capture": "tasks.capture.operation",
    "intervene": "tasks.intervene.operation",
    "generate": "tasks.generate.operation",
    "circuits": "tasks.circuits.operation",
    "explain": "tasks.explain.operation",
    "benchmark": "tasks.benchmark.operation",
    "train": "tasks.train.operation",
}


def execute_task(
    operation: str,
    cfg: Any,
    *,
    generator: Callable[[list[str]], str] | None = None,
    predictor: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Dispatch to the task-owned execution module after selection checks."""
    from importlib import import_module

    _require_context(cfg, operation)
    if not bool(_task(cfg, "execute", False)):
        return {
            "status": "configured_only",
            "operation": operation,
            "model_key": str(cfg.model.name),
            "dataset_key": str(cfg.dataset.name),
            "description": "Set task.execute=true and supply required runtime inputs to execute.",
        }
    module = import_module(RUNNER_MODULES[operation])
    if operation == "explain":
        result = module.execute(cfg, generator=generator, predictor=predictor)
    else:
        result = module.execute(cfg)
    return _maybe_save(operation, result)


class SAEOperationTask(TaskModule):
    """Runtime-neutral task adapter; all algorithms live in their task package."""

    OPERATION = ""

    @property
    def name(self) -> str:
        return self.OPERATION

    def run(self, cfg: Any) -> dict[str, Any]:
        return execute_task(self.OPERATION, cfg)
