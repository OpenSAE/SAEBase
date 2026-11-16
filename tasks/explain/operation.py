"""Concrete explain task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

import json
from typing import Any, Callable

from tasks.sae_runtime import _task, input_path


def _explain(
    cfg: Any,
    generator: Callable[[list[str]], str] | None,
    predictor: Callable[[str, str], bool] | None,
) -> dict[str, Any]:
    from tasks.explain.interpretation import propose_interpretation

    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Auto-explanation needs task.approved=true")
    if generator is None:
        generator = _allowlisted_generator(cfg)
    if generator is None:
        raise RuntimeError(
            "Supply a real inference provider with ExplainTask(generator=callable) "
            "or task.generator=saebase_generate. "
            "The task will not invent an explanation or call a paid service implicitly."
        )
    discovery_path = input_path(_task(cfg, "discovery_file", ""))
    heldout_path = input_path(_task(cfg, "heldout_file", ""))
    if not discovery_path.is_file() or not heldout_path.is_file():
        raise FileNotFoundError("Provide actual discovery_file and heldout_file")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    response = propose_interpretation(
        site_id=str(_task(cfg, "site_id", "")),
        feature_id=int(_task(cfg, "feature_id", 0)),
        discovery=discovery,
        heldout=heldout,
        generator=generator,
        predictor=predictor,
        generator_model=str(_task(cfg, "generator_model", "")),
        generator_method=str(_task(cfg, "generator_method", "")),
        authorized_budget_usd=float(_task(cfg, "authorized_budget_usd", 0)),
        estimated_cost_usd=float(_task(cfg, "estimated_cost_usd", 0)),
    )
    from saebase.results import completion_summary

    return {
        "completion": completion_summary([{"status": "completed"}]),
        "status": "executed",
        "operation": "explain",
        "model_key": str(cfg.model.name),
        "hypothesis": response,
    }


def _allowlisted_generator(cfg: Any) -> Callable[[list[str]], str] | None:
    """Resolve one named local generator. Arbitrary strings are not code."""
    name = str(_task(cfg, "generator", "") or "").strip()
    if not name:
        return None
    if name != "saebase_generate":
        raise ValueError(
            "Unknown explain generator. The only allowlisted local implementation "
            "is saebase_generate."
        )
    from tasks.explain.local_generator import saebase_generate

    return saebase_generate(cfg)


def execute(cfg: Any, generator=None, predictor=None) -> dict[str, Any]:
    return _explain(cfg, generator, predictor)


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from tasks.sae_runtime import validate_fields, validate_native_selection

    validate_native_selection(cfg)
    validate_fields(
        cfg,
        integers=(("feature_id", 0), ("generator_max_new_tokens", 1)),
        numbers=(("authorized_budget_usd", 0), ("estimated_cost_usd", 0)),
        booleans=("approved",),
    )
    if cfg.task.get("generator", "") not in ("", "saebase_generate"):
        raise ValueError("Unsupported explanation generator")
    if cfg.task.get("estimated_cost_usd", 0) > cfg.task.get("authorized_budget_usd", 0):
        raise ValueError("Explanation cost exceeds its authorized budget")
    if cfg.task.get("execute", False):
        for field in ("discovery_file", "heldout_file"):
            if not input_path(cfg.task.get(field, "")).is_file():
                raise FileNotFoundError(f"Provide an existing task.{field}")


def prepare_execution(cfg: Any) -> dict:
    """Prepare the selected protocol without running the research calculation."""
    generator = _allowlisted_generator(cfg)
    if generator is None:
        raise ValueError(
            "Configure task.generator=saebase_generate or supply ExplainTask(generator=callable)"
        )
    for field in ("discovery_file", "heldout_file"):
        rows = json.loads(input_path(cfg.task[field]).read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"task.{field} must contain nonempty example records")
    return {
        "status": "PASS",
        "scope": "explanation inputs and generator configured; no generation or scoring executed",
    }
