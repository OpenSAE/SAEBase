"""可选的 vLLM HTTP 服务。默认不另起进程：当前解释器里有 vLLM 就直接用。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tasks.LLM.generate.backend.settings import (
    DTYPE,
    GPU_MEMORY_UTILIZATION,
    HOST,
    MAX_MODEL_LEN,
    PORT,
    apply_process_env,
    repo_root,
)

_PROC: subprocess.Popen | None = None


def stop_owned_server(*, timeout_s: float = 5.0) -> bool:
    """Stop the subprocess launched by this Python interpreter, if any.

    Return True when this call owned a running process. A reused external
    server is never tracked in ``_PROC`` and is never terminated.
    """
    global _PROC
    proc, _PROC = _PROC, None
    if proc is None or proc.poll() is not None:
        return False
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)
    return True


def vllm_python() -> Path:
    return Path(sys.executable)


def server_port() -> int:
    return int(os.environ.get("VLLM_PORT", str(PORT)))


def server_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", f"http://127.0.0.1:{server_port()}/v1").rstrip("/")


def autostart_enabled() -> bool:
    return os.environ.get("SAE_GENERATE_AUTOSTART", "1").strip() not in {"0", "false", "no"}


def server_healthy(timeout_s: float = 2.0) -> bool:
    url = server_base_url() + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return int(resp.status) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ensure_server(*, model: str) -> None:
    """已经有服务就复用。否则用当前解释器拉起，不再找另一个虚拟环境。"""
    if server_healthy():
        return
    if not autostart_enabled():
        raise RuntimeError(
            "vLLM is not in this interpreter and no server is listening at "
            f"{server_base_url()}. Install vllm into this interpreter, or set "
            "SAE_GENERATE_AUTOSTART=1."
        )
    apply_process_env()
    py = vllm_python()
    log_dir = repo_root() / "experiment" / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "vllm_server.log"
    env = os.environ.copy()
    cmd = [
        str(py),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--host",
        HOST,
        "--port",
        str(server_port()),
        "--dtype",
        DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--enable-prefix-caching",
    ]
    global _PROC
    log_f = log_path.open("ab")
    try:
        _PROC = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    finally:
        log_f.close()
    deadline = time.time() + float(os.environ.get("VLLM_SERVER_WAIT_S", "180"))
    while time.time() < deadline:
        if server_healthy(timeout_s=1.0):
            return
        if _PROC.poll() is not None:
            _PROC = None
            raise RuntimeError(
                f"vLLM server exited early (code {_PROC.returncode}). See {log_path}."
            )
        time.sleep(1.0)
    stop_owned_server()
    raise RuntimeError(
        f"vLLM server did not become healthy at {server_base_url()}. See {log_path}."
    )


def post_completions(body: dict) -> dict:
    url = server_base_url() + "/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    timeout = float(os.environ.get("VLLM_HTTP_TIMEOUT_S", "7200"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
