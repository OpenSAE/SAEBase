"""Generic immutable run-artifact integrity audit.

The owning Dataset/Model/Task evaluator still verifies scientific semantics;
this module verifies declared file identities, completeness and block scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SHA = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    """Hash actual file bytes without loading an entire checkpoint in memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def audit_artifacts(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check an immutable manifest with hard/optional failures kept separate."""
    if manifest.get("schema") != "saebase.artifact_manifest.v1":
        raise ValueError("Unsupported artifact manifest schema")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected at least one declared artifact")
    root = root.resolve(strict=True)
    seen = set()
    defects = []
    ok = 0
    failed_optional = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed artifact entry")
        path = record.get("path")
        sha = record.get("sha256")
        size = record.get("bytes")
        required = record.get("required", True)
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or PurePosixPath(path).is_absolute()
            or any(x in ("", ".", "..") for x in path.split("/"))
            or path in seen
            or type(size) is not int
            or size < 0
            or not isinstance(sha, str)
            or SHA.fullmatch(sha) is None
            or type(required) is not bool
        ):
            raise ValueError("Malformed, repeated or escaping artifact identity")
        seen.add(path)
        actual = (root / path).resolve(strict=False)
        if not actual.is_relative_to(root):
            raise ValueError("Artifact path or symlink escaped the run root")
        reason = None
        if not actual.is_file():
            reason = "missing"
        elif actual.stat().st_size != size or file_sha256(actual) != sha:
            reason = "bytes_or_sha256_mismatch"
        if reason:
            defects.append({"path": path, "reason": reason, "required": required})
            if not required:
                failed_optional += 1
        else:
            ok += 1
    hard = any(r["required"] for r in defects)
    max_optional = manifest.get("max_optional_failures", 0)
    if type(max_optional) is not int or max_optional < 0:
        raise ValueError("Invalid optional failure threshold")
    return {
        "schema": "saebase.artifact_audit.v1",
        "status": "BLOCKED"
        if hard or failed_optional > max_optional
        else ("PARTIAL" if defects else "PASS"),
        "verified_files": ok,
        "declared_files": len(records),
        "defects": defects,
        "scientific_validation": "NOT_COVERED",
        "independent_harness_acceptance": "NOT_COVERED",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit_artifacts(args.root, json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"status": report["status"], "receipt": str(args.output)}))
    return 0 if report["status"] != "BLOCKED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
