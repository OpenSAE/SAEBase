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
    return {
        "status": "executed",
        "operation": "circuits",
        "model_key": str(cfg.model.name),
        "prompt_sha256": _fingerprint_prompt(prompt),
        "native": details,
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _circuits(cfg)
