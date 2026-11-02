"""Prepare dependencies and cached Hub assets; never execute a scientific test.

Every package installation or download is opt-in and budget-bound. A ready
asset is not a model-load/capture PASS, which only Precheck can report.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _issue(code: str, reason: str, attempted: list[str], repair: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "code": code,
        "reason": reason,
        "attempted": attempted,
        "repair": repair,
        "stage": "setup",
    }


def prepare_environment(
    *,
    required_modules: tuple[str, ...] = ("transformer_lens", "sae_lens", "huggingface_hub"),
    assets: list[dict[str, str | None]] | None = None,
    allow_network: bool = False,
    allow_install: bool = False,
    allow_download: bool = False,
    max_download_bytes: int = 0,
    install_timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Stage only requested packages/checkpoints, never loading a Model or SAE.

    A cached repo may be used without network access. Network downloads have
    an explicit byte ceiling measured from Hub file metadata before transfer.
    """
    requested = assets or []
    attempted: list[str] = []
    if type(max_download_bytes) is not int or max_download_bytes < 0:
        raise ValueError("max_download_bytes must be a nonnegative integer")
    missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
    if missing and not (allow_network and allow_install):
        return _issue(
            "BLOCKED_DEPENDENCY",
            f"Missing Python modules: {missing}",
            attempted,
            "Authorize Setup dependency installation (allow_install + allow_network), or install the SAEBase analysis extra.",
        )
    if missing:
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-m", "pip", "install", "-e", ".[analysis]"]
        attempted.append("pip install -e .[analysis]")
        try:
            proc = subprocess.run(
                command, cwd=root, capture_output=True, text=True,
                timeout=install_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            return _issue(
                "BLOCKED_DEPENDENCY", "Dependency installation exceeded the allowed time",
                attempted, "Review the compatible Python/Torch environment and retry Setup.",
            )
        if proc.returncode:
            return _issue(
                "BLOCKED_DEPENDENCY",
                f"Dependency installation failed (exit {proc.returncode}): {proc.stderr[-900:]}",
                attempted, "Repair dependency resolution without silently changing the scientific protocol.",
            )
        importlib.invalidate_caches()
        missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
        if missing:
            return _issue(
                "BLOCKED_DEPENDENCY", f"Modules remain unavailable after installation: {missing}",
                attempted, "Use the correct Python environment and rerun Setup.",
            )
    prepared = []
    if requested:
        if importlib.util.find_spec("huggingface_hub") is None:
            return _issue(
                "BLOCKED_DEPENDENCY", "huggingface_hub is unavailable", attempted,
                "Install the analysis extra through Setup with explicit authorization.",
            )
        from huggingface_hub import HfApi, snapshot_download

        remaining = max_download_bytes
        for item in requested:
            repo_id = str(item.get("repo_id") or "").strip()
            revision = item.get("revision")
            if not repo_id or "/" not in repo_id:
                return _issue(
                    "BLOCKED_CONFIGURATION", f"Invalid Hub repository identity: {repo_id!r}",
                    attempted, "Specify a real model or SAE repository and optionally its immutable revision.",
                )
            try:
                cache_path = snapshot_download(
                    repo_id=repo_id, revision=revision, local_files_only=True,
                )
                prepared.append({"repo_id": repo_id, "revision": revision, "cache_path": cache_path, "source": "cache"})
                continue
            except (OSError, ValueError):
                pass
            if not (allow_network and allow_download):
                return _issue(
                    "BLOCKED_ASSET", f"Repository {repo_id} is not fully cached",
                    attempted,
                    "Authorize network asset preparation (allow_download + allow_network), or copy the pinned assets into the Hub cache.",
                )
            try:
                info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
                sizes = [getattr(f, "size", None) for f in info.siblings or ()]
                if not sizes or any(type(n) is not int or n < 0 for n in sizes):
                    return _issue(
                        "BLOCKED_RESOURCE", f"Could not determine complete download size for {repo_id}",
                        attempted, "Provide a pinned, size-verifiable asset source before downloading.",
                    )
                needed = sum(sizes)
                if needed > remaining:
                    return _issue(
                        "BLOCKED_RESOURCE", f"{repo_id} needs {needed} bytes; remaining permitted budget is {remaining}",
                        attempted, "Increase the explicit Setup download budget or provide pre-cached assets.",
                    )
                attempted.append(f"snapshot_download {repo_id}@{revision or 'resolved'}")
                cache_path = snapshot_download(repo_id=repo_id, revision=revision)
                remaining -= needed
                prepared.append({
                    "repo_id": repo_id, "revision": getattr(info, "sha", revision),
                    "cache_path": cache_path, "source": "download",
                })
            except Exception as exc:
                return _issue(
                    "BLOCKED_ASSET",
                    f"{repo_id}: {type(exc).__name__}: {str(exc)[:650]}",
                    attempted, "Check Hub access, repository revision, network credentials and disk; retry without replacing this asset.",
                )
    return {
        "status": "READY",
        "stage": "setup",
        "prepared_assets": prepared,
        "required_modules": list(required_modules),
        "attempted": attempted,
        "scientific_test": "NOT_RUN",
        "gpu_test": "NOT_RUN",
    }
