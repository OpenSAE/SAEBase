"""Concrete circuits task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tasks.LLM.sae_runtime import MODEL_KEY, _task, _prompt, _fingerprint_prompt


def _circuits(cfg: Any) -> dict[str, Any]:
    from tasks.LLM.circuits.backend.circuits import run_native_circuit_tracer

    if not bool(_task(cfg, "approved", False)):
        raise PermissionError("Circuit tracing requires task.approved=true")
    prompt = _prompt(cfg)
    destination = Path.cwd() / "figures" / "circuit_trace"
    details = run_native_circuit_tracer(
        prompt=prompt,
        transcoder_set=str(_task(cfg, "transcoder_set", "gemma")),
        output=destination,
        slug=str(_task(cfg, "slug", "gemma2b-trace")),
        approved=True,
        timeout_seconds=int(_task(cfg, "timeout_seconds", 1800)),
    )
    return {
        "status": "executed",
        "operation": "circuits",
        "model_key": MODEL_KEY,
        "prompt_sha256": _fingerprint_prompt(prompt),
        "native": details,
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _circuits(cfg)
