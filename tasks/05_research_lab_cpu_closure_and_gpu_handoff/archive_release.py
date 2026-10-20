"""Private archive of the current release. Weights stay in place."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from common import REPO, file_sha256, free_gib, write_json


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()


def archive_release(root: Path) -> dict:
    """Copy measurement files and record SHA-256. Do not duplicate the weight cache."""
    dest = root.parent / "archive" / root.name
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "GPU-02" / "decision.json").is_file():
        for child in root.iterdir():
            if child.name in {"cache", "tmp"}:
                continue
            target = dest / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
    rows = []
    for path in sorted(item for item in dest.rglob("*") if item.is_file() and item.name != "manifest.json"):
        rows.append(
            {
                "path": str(path.relative_to(dest)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    cache = root / "cache"
    cache_rows = []
    cache_bytes = 0
    if cache.is_dir():
        for path in sorted(item for item in cache.rglob("*") if item.is_file()):
            size = path.stat().st_size
            cache_bytes += size
            cache_rows.append({"path": str(path.relative_to(root)), "bytes": size})
    try:
        status = _git(["status", "--porcelain"])
        commit = _git(["rev-parse", "HEAD"])
    except subprocess.CalledProcessError as exc:
        status = exc.output
        commit = "unknown"
    payload = {
        "release": root.name,
        "runner_commit": commit,
        "dirty": bool(status),
        "git_status": status,
        "archive_copy": str(dest),
        "copied_files": len(rows),
        "copied_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "weight_cache": {
            "copied": False,
            "bytes": cache_bytes,
            "files": len(cache_rows),
            "reason": "A second full weight copy would consume the free space needed to keep the original cache.",
            "inventory": cache_rows,
        },
        "free_gib_at_archive": round(free_gib(root), 2),
    }
    digest = hashlib.sha256(
        json.dumps(
            {"commit": commit, "files": [(row["path"], row["sha256"]) for row in rows]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload["manifest_sha256"] = digest
    write_json(dest / "manifest.json", payload)
    write_json(root / "GPU-00" / "archive_manifest.json", {k: v for k, v in payload.items() if k != "files"})
    pointer = root / "GPU-00" / "archive_file_index.json"
    write_json(pointer, {"manifest_sha256": digest, "archive_copy": str(dest), "copied_files": len(rows)})
    os.sync()
    return payload
