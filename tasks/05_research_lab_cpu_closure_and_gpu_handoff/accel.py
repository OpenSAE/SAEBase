"""Deployment acceleration that does not change the estimators.

vLLM is probed in this interpreter and installed only when the resolve keeps
the current torch build. SAE capture, ablation and SAR stay on the hook API.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path

from common import file_sha256, log, write_json

SAR_REPO = "https://github.com/OpenSAE/SAE-Behavioral-Compositionality.git"
RSS_REPO = "https://github.com/OpenSAE/Redistribution-Sensitivity-Score.git"


def _clone(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    return commit


def prepare_estimators(root: Path) -> dict:
    originals = root / "originals"
    sar_dir = originals / "SAR"
    rss_dir = originals / "RSS"
    sar_commit = _clone(SAR_REPO, sar_dir)
    rss_commit = _clone(RSS_REPO, rss_dir)
    sar_file = next(sar_dir.rglob("sar.py"))
    rss_file = next(rss_dir.rglob("rss.py"))
    payload = {
        "sar_commit": sar_commit,
        "rss_commit": rss_commit,
        "sar_py": str(sar_file),
        "rss_py": str(rss_file),
        "sar_sha256": file_sha256(sar_file),
        "rss_sha256": file_sha256(rss_file),
    }
    write_json(root / "GPU-00" / "estimator_sources.json", payload)
    log(root, f"original estimators sar={sar_commit[:12]} rss={rss_commit[:12]}")
    return payload


def probe_vllm(root: Path) -> dict:
    """Refuse a vLLM install that would replace the registered torch build."""
    torch_version = importlib.metadata.version("torch")
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    probe = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "vllm"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    text = (probe.stdout or "") + "\n" + (probe.stderr or "")
    mentions_torch_removal = "uninstall" in text.lower() and "torch" in text.lower()
    confirmed_same_torch = (not mentions_torch_removal) and any(
        torch_version in line
        and "torch" in line.lower()
        and ("install" in line.lower() or line.strip().startswith("+"))
        for line in text.splitlines()
    )
    payload = {
        "torch": torch_version,
        "vllm_installed": installed,
        "dry_run_returncode": probe.returncode,
        "would_replace_torch": not confirmed_same_torch,
        "decision": "not_installed",
    }
    if installed:
        payload["decision"] = "already_present_generation_only"
    elif probe.returncode == 0 and confirmed_same_torch:
        install = subprocess.run(
            ["uv", "pip", "install", "vllm"], capture_output=True, text=True, timeout=1800
        )
        payload["install_returncode"] = install.returncode
        if install.returncode == 0 and importlib.metadata.version("torch") == torch_version:
            payload["decision"] = "installed_same_torch"
            payload["vllm_installed"] = importlib.metadata.version("vllm")
        else:
            payload["decision"] = "install_failed_full_prefix_retained"
            payload["install_tail"] = (install.stderr or install.stdout or "")[-500:]
    else:
        payload["decision"] = "skipped_to_protect_torch"
        payload["note"] = (
            "Pinned vllm==0.27.1 replaces torch 2.11+cu128. "
            "Generation stays on full-prefix until a same-torch wheel exists. "
            "No KV-cache parity is claimed."
        )
    write_json(root / "GPU-00" / "vllm_probe.json", payload)
    log(root, "vLLM probe " + payload["decision"])
    return payload
