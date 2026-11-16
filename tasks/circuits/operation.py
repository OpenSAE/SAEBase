"""Concrete circuits task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tasks.sae_runtime import _task, _prompt, _fingerprint_prompt


def _circuits(cfg: Any) -> dict[str, Any]:
    from tasks.circuits.backend.circuits import run_native_circuit_tracer

    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Circuit tracing requires task.approved=true")
    prompt = _prompt(cfg)
    destination = Path.cwd() / "figures" / "circuit_trace"
    feature_cap = _task(cfg, "max_feature_nodes", None)
    backward_batch = _task(cfg, "batch_size", None)
    offload = _task(cfg, "offload", None)
    dtype = _task(cfg, "dtype", None)
    details = run_native_circuit_tracer(
        prompt=prompt,
        transcoder_set=str(_task(cfg, "transcoder_set", "gemma")),
        output=destination,
        slug=str(_task(cfg, "slug", "gemma2b-trace")),
        approved=True,
        timeout_seconds=int(_task(cfg, "timeout_seconds", 1800)),
        offload=None if offload in (None, "") else str(offload),
        dtype=None if dtype in (None, "") else str(dtype),
        max_feature_nodes=None if feature_cap in (None, "") else int(feature_cap),
        batch_size=None if backward_batch in (None, "") else int(backward_batch),
        lazy_encoder=bool(_task(cfg, "lazy_encoder", False)),
    )
    from saebase.results import completion_summary

    return {
        "completion": completion_summary([{"status": "completed"}]),
        "status": "executed",
        "operation": "circuits",
        "input_scope": {
            "unit": "prompt",
            "requested": 1,
            "processed": 1,
            "source": "explicit_prompt" if cfg.task.get("prompt") else "first_dataset_record",
        },
        "model_key": str(cfg.model.name),
        "prompt_sha256": _fingerprint_prompt(prompt),
        "native": details,
        "resolved_protocol": {
            "launch_labels": {
                "model": str(cfg.model.name),
                "dataset": str(getattr(getattr(cfg, "dataset", None), "name", "")),
            },
            "transcoder_set": str(_task(cfg, "transcoder_set", "gemma")),
            "model_identity": "requires_native_source_provenance",
        },
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _circuits(cfg)


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from tasks.sae_runtime import validate_fields, validate_native_selection

    validate_native_selection(cfg)
    validate_fields(
        cfg,
        integers=(("timeout_seconds", 1), ("max_feature_nodes", 1), ("batch_size", 1)),
        booleans=("approved", "lazy_encoder"),
    )
    if cfg.task.get("execute", False):
        from tasks.circuits.backend.circuits import run_native_circuit_tracer

        run_native_circuit_tracer(
            prompt=str(cfg.task.get("prompt") or "validation"),
            transcoder_set=str(cfg.task.get("transcoder_set", "gemma")),
            output=Path("unused_validation_output"),
            slug=str(cfg.task.get("slug", "gemma2b-trace")),
            timeout_seconds=cfg.task.get("timeout_seconds", 1800),
            offload=cfg.task.get("offload"),
            dtype=cfg.task.get("dtype"),
            max_feature_nodes=cfg.task.get("max_feature_nodes"),
            batch_size=cfg.task.get("batch_size"),
            validate_only=True,
        )


def prepare_execution(cfg: Any) -> dict:
    """Prepare the selected protocol without running the research calculation."""
    import shutil

    if shutil.which("circuit-tracer") is None:
        raise FileNotFoundError("Install circuit-tracer in the selected worker environment")
    return {
        "status": "PASS",
        "scope": "circuit-tracer inputs and executable ready; no attribution executed",
    }
