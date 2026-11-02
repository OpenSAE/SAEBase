"""Local-only immutable delivery after a PASS evidence audit.

This module never uploads to B2, deletes sources or copies model checkpoints
unless such files were individually declared in the authorized manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

from saebase.runtime.audit_report import audit_run, _source


def deliver(
    run_root: Path,
    manifest: dict[str, Any],
    archive_dir: Path,
    *,
    archive: bool = True,
) -> dict[str, Any]:
    report = audit_run(run_root, manifest)
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not all(c.isalnum() or c in "_-" for c in run_id)
    ):
        raise ValueError("invalid run ID")
    result: dict[str, Any] = {
        "schema": "saebase.audit_delivery.v1",
        "run_id": run_id,
        "evidence_status": report["evidence_status"],
        "execution_status": report["execution_status"],
        "archive_status": "BLOCKED",
        "remote_backup": "NOT_AUTHORIZED",
    }
    if report["evidence_status"] != "PASS":
        result["reason"] = "evidence audit did not pass; no archive created"
        return result
    if not archive:
        result["archive_status"] = "SKIPPED"
        return result
    archive_dir.mkdir(parents=True, exist_ok=True)
    output = archive_dir / f"{run_id}.tar.gz"
    if output.exists():
        result["archive_status"] = "BLOCKED"
        result["reason"] = "immutable archive name already exists"
        return result
    files = [record["path"] for record in manifest["files"]]
    # Restrict to declared, hashed evidence. Symlinks are never archived.
    if any(_source(run_root.resolve(), name).is_symlink() for name in files):
        result["reason"] = "symlink evidence cannot be archived"
        return result
    try:
        with output.open("xb") as stream:
            with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
                for name in files:
                    path = _source(run_root.resolve(), name)
                    if not path.is_file():
                        raise ValueError("declared file disappeared")
                    bundle.add(path, arcname=name, recursive=False)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        result.update(
            archive_status="ARCHIVED", archive_path=str(output), sha256=digest, files=files
        )
    except (OSError, ValueError):
        output.unlink(missing_ok=True)
        result["reason"] = "archive creation failed"
    return result


def write_delivery(receipt: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
