from __future__ import annotations

import os
import subprocess
from pathlib import Path

from src.schema import require_keys

DATE = "2001-01-01T00:00:00+0000"


def _git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(root),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_DATE": DATE,
            "GIT_COMMITTER_DATE": DATE,
        },
    )
    if result.returncode:
        raise ValueError("git failed")
    return result.stdout.strip()


def execute(payload, workdir: Path):
    require_keys(payload, {"base", "a", "b", "resolution", "resume_from"})
    root = Path(workdir)
    _git(root, "init")
    _git(root, "config", "user.email", "exam@example.invalid")
    _git(root, "config", "user.name", "Synthetic Exam")
    target = root / "module.py"
    if not all(isinstance(payload[key], str) for key in ("base", "a", "b")):
        raise ValueError("invalid source")
    target.write_text(payload["base"], encoding="utf-8")
    _git(root, "add", "module.py")
    _git(root, "commit", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    _git(root, "switch", "-c", "agent-a")
    target.write_text(payload["a"], encoding="utf-8")
    _git(root, "add", "module.py")
    if payload["a"] == payload["base"]:
        raise ValueError("agent A made no change")
    _git(root, "commit", "-m", "agent-a")
    head_a = _git(root, "rev-parse", "HEAD")
    _git(root, "switch", "-c", "agent-b", base)
    target.write_text(payload["b"], encoding="utf-8")
    _git(root, "add", "module.py")
    if payload["b"] == payload["base"]:
        raise ValueError("agent B made no change")
    _git(root, "commit", "-m", "agent-b")
    head_b = _git(root, "rev-parse", "HEAD")
    if payload["resume_from"] != "HEAD_AFTER_A":
        raise ValueError("stale continuation")
    _git(root, "switch", "-c", "merged")
    merged = subprocess.run(
        ["git", "-C", str(root), "merge", "--no-ff", "agent-a", "-m", "merge"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(root),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_DATE": DATE,
            "GIT_COMMITTER_DATE": DATE,
        },
    )
    conflicted = merged.returncode != 0
    if conflicted:
        if not isinstance(payload["resolution"], str) or not payload["resolution"]:
            raise ValueError("RESOLUTION_REQUIRED")
        target.write_text(payload["resolution"], encoding="utf-8")
        _git(root, "add", "module.py")
        _git(root, "commit", "-m", "resolved-conflict")
    elif payload["resolution"] is not None:
        raise ValueError("Unexpected manual resolution without conflict")
    merged_head = _git(root, "rev-parse", "HEAD")
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if set(parents[1:]) != {head_a, head_b}:
        raise ValueError("Merge parents missing")
    return {
        "base_sha": base,
        "head_a_sha": head_a,
        "head_b_sha": head_b,
        "merge_sha": merged_head,
        "merge_parents": parents[1:],
        "commits": int(_git(root, "rev-list", "--count", "--all")),
        "conflicted": conflicted,
        "source": target.read_text(encoding="utf-8"),
    }
