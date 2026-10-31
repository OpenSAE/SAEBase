"""Isolated, bounded execution of registered experiments on heterogeneous machines.

A failed CUDA worker is allowed one fresh-process retry at identical
scientific settings. This avoids carrying a potentially corrupted CUDA
allocator into the next attempt. Unsupported precision changes or alternative
models are never chosen as OOM workarounds.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_worker(
    *,
    argv: list[str],
    receipt_file: Path,
    log_dir: Path,
    retry_limit: int = 1,
    timeout_seconds: int = 172800,
) -> dict[str, Any]:
    if not argv or retry_limit < 0 or retry_limit > 3:
        raise ValueError("Invalid isolated experiment worker specification")
    if timeout_seconds < 30 or timeout_seconds > 172800:
        raise ValueError("Unapproved worker timeout")
    log_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for index in range(retry_limit + 1):
        started = _utc()
        stdout = log_dir / f"attempt_{index:02d}.stdout.log"
        stderr = log_dir / f"attempt_{index:02d}.stderr.log"
        error_type = None
        try:
            with stdout.open("w", encoding="utf-8") as so, stderr.open("w", encoding="utf-8") as se:
                proc = subprocess.run(
                    argv,
                    stdout=so,
                    stderr=se,
                    check=False,
                    timeout=timeout_seconds,
                )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            error_type = "TIMEOUT"
        except OSError as exc:
            code = 127
            error_type = type(exc).__name__
            stderr.write_text(str(exc)[:500], encoding="utf-8")
        content = ""
        if stderr.exists():
            content += stderr.read_text(encoding="utf-8")[-4000:]
        if stdout.exists():
            content += stdout.read_text(encoding="utf-8")[-4000:]
        # Exact error classification can also be recovered from a per-model
        # receipt, which remains authoritative when a child crashes mid-run.
        if receipt_file.exists():
            try:
                prior = json.loads(receipt_file.read_text(encoding="utf-8"))
                for step in prior.get("steps", []):
                    content += str(step.get("error_type", "")) + str(step.get("reason", ""))
            except (ValueError, OSError):
                pass
        oom = "OutOfMemoryError" in content or "CUDA out of memory" in content
        row = {
            "index": index,
            "argv": argv,
            "started_utc": started,
            "ended_utc": _utc(),
            "returncode": code,
            "stdout": str(stdout),
            "stderr": str(stderr),
            "reason": error_type or ("CUDA_OOM" if oom else "PROCESS_EXIT"),
        }
        attempts.append(row)
        if code == 0:
            break
        if not oom or index >= retry_limit:
            break
    result = {
        "schema": "saebase.execution.worker_attempts.v1",
        "status": "completed"
        if attempts[-1]["returncode"] == 0
        else ("blocked_resource" if attempts[-1]["reason"] == "CUDA_OOM" else "failed"),
        "attempts": attempts,
        "recovery": "fresh_process_same_model_sae_dtype_and_inputs_only",
        "precision_changed": False,
        "model_substituted": False,
    }
    output = log_dir / "attempts.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != result:
            # A new invocation is an independent attempt; never overwrite
            # prior evidence with a different exit code or timestamp.
            numbered = len(list(log_dir.glob("prior_attempts_*.json")))
            output.rename(log_dir / f"prior_attempts_{numbered:04d}.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
