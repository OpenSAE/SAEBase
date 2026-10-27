"""用当前解释器启动 NNsight worker。不另找一套 Python。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Sequence

from tasks.LLM.generate.backend.settings import apply_process_env, repo_root


def worker_cmd(argv: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "tasks.LLM.generate.backend.nnsight_worker", *list(argv)]


def _child_env() -> dict[str, str]:
    apply_process_env()
    env = os.environ.copy()
    root = str(repo_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _verify_available_worker_mode(argv: Sequence[str]) -> None:
    """Fail fast when archived experiment-only code is missing."""
    if argv and argv[0] == "generate":
        missing = [
            rel for rel in (
                "tasks/10_cross_model_steering/gemmascope_rows.py",
                "tasks/13_functional_concept_validation/sae_edit.py",
            ) if not (repo_root() / rel).is_file()
        ]
        if missing:
            raise RuntimeError(
                "NNsight protocol generation requires source modules that were "
                "not shipped in standalone SAEBase: " + ", ".join(missing) + ". "
                "Use the original research workspace, or run the independently "
                "supported smoke/probe/dump_resid commands."
            )


def _release_parent_gpu() -> None:
    from tasks.LLM.generate.backend.gpu import free_cuda, stop_vllm

    stop_vllm()
    free_cuda()


def _payload_from_text(text: str, returncode: int) -> dict[str, Any]:
    last = ""
    for line in text.strip().splitlines()[::-1]:
        if line.lstrip().startswith("{"):
            last = line
            break
    if last:
        try:
            payload = json.loads(last)
        except json.JSONDecodeError:
            payload = {"ok": False, "error": "bad json from worker", "raw": text[-2000:]}
    else:
        payload = {"ok": False, "error": "no json from worker", "raw": text[-2000:]}
    payload["returncode"] = returncode
    if returncode != 0 and not payload.get("error"):
        payload["error"] = text[-2000:]
        payload["ok"] = False
    return payload


def run_worker(argv: Sequence[str], *, timeout: int = 1800) -> dict[str, Any]:
    """跑完再返回。最后一行 JSON 是结果。"""
    _verify_available_worker_mode(argv)
    _release_parent_gpu()
    proc = subprocess.run(
        worker_cmd(argv),
        env=_child_env(),
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        timeout=int(timeout),
        check=False,
    )
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    payload = _payload_from_text(proc.stdout or "", proc.returncode)
    if proc.returncode != 0 and not payload.get("error"):
        payload["error"] = text[-2000:]
        payload["ok"] = False
    payload["stderr_tail"] = (proc.stderr or "")[-4000:]
    return payload


def run_worker_stream(argv: Sequence[str], *, timeout: float = 7200) -> dict[str, Any]:
    """Stream worker output with a wall-clock deadline, even when it is silent."""
    import queue
    import threading
    import time

    _verify_available_worker_mode(argv)
    _release_parent_gpu()
    proc = subprocess.Popen(
        worker_cmd(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=_child_env(), cwd=str(repo_root()), bufsize=1,
    )
    assert proc.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    chunks: list[str] = []
    deadline = time.monotonic() + float(timeout)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"nnsight worker exceeded {timeout}s")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"nnsight worker exceeded {timeout}s") from None
            if line is None:
                break
            print(line, end="", flush=True)
            chunks.append(line)
        rc = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
    except (TimeoutError, subprocess.TimeoutExpired):
        proc.kill()
        proc.wait()
        raise TimeoutError(f"nnsight worker exceeded {timeout}s") from None
    finally:
        proc.stdout.close()
        thread.join(timeout=1.0)
    payload = _payload_from_text("".join(chunks), rc)
    if rc != 0 and not payload.get("error"):
        payload["error"] = "".join(chunks)[-2000:]
        payload["ok"] = False
    return payload
