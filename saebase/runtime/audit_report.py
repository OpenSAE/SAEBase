"""Run-wide evidence audit: checks receipts, not scientific success.

The manifest is explicit: the caller must enumerate requested checks and report
paths. Never infer PASS from the presence of a downloaded/configured resource.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from saebase.runtime.audit import audit_artifacts

STATES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_RUN"})
EXPECTED_KINDS = frozenset({"setup", "precheck", "worker", "task", "workplan"})


def _source(root: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise ValueError("invalid receipt path")
    actual = (root / name).resolve(strict=False)
    if not actual.is_relative_to(root.resolve()):
        raise ValueError("receipt escapes run root")
    return actual


def audit_run(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Evidence completeness can PASS even if real execution FAILS/BLOCKS."""
    root = root.resolve(strict=True)
    files = audit_artifacts(root, manifest)
    checks = manifest.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks must explicitly enumerate the requested targets")
    seen = set()
    rows: list[dict[str, Any]] = []
    defects: list[dict[str, str]] = []
    for item in checks:
        if not isinstance(item, dict):
            raise ValueError("invalid requested check")
        cid, kind, path = item.get("id"), item.get("kind"), item.get("receipt")
        if not isinstance(cid, str) or not cid or cid in seen or kind not in EXPECTED_KINDS:
            raise ValueError("invalid or repeated check identity")
        seen.add(cid)
        row: dict[str, Any] = {"id": cid, "kind": kind, "receipt": path, "status": "MISSING"}
        try:
            actual = _source(root, path)
            if not actual.is_file():
                raise ValueError("receipt missing")
            if actual.stat().st_size > 2_000_000:
                raise ValueError("receipt too large")
            receipt = json.loads(actual.read_text(encoding="utf-8"))
            state = receipt.get("status")
            if state not in STATES:
                raise ValueError("invalid receipt status")
            if receipt.get("run_id") != manifest.get("run_id"):
                raise ValueError("run identity mismatch")
            row["status"] = state
            row["source_sha"] = receipt.get("source_sha")
            if manifest.get("source_sha") and receipt.get("source_sha") != manifest.get(
                "source_sha"
            ):
                raise ValueError("source identity mismatch")
            if path not in {x.get("path") for x in manifest["files"]}:
                raise ValueError("receipt omitted from artifact manifest")
        except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
            row["status"] = "INVALID"
            defects.append({"id": cid, "reason": str(exc)[:250]})
        rows.append(row)
    if not checks:
        defects.append({"id": "manifest", "reason": "no requested checks declared"})
    artifact_status = files["status"]
    evidence_status = "PASS" if artifact_status == "PASS" and not defects else "BLOCKED"
    return {
        "schema": "saebase.run_audit.v1",
        "run_id": manifest.get("run_id"),
        "source_sha": manifest.get("source_sha"),
        "evidence_status": evidence_status,
        "artifact_status": artifact_status,
        "execution_status": (
            "FAIL"
            if any(x["status"] == "FAIL" for x in rows)
            else "BLOCKED"
            if any(x["status"] == "BLOCKED" for x in rows)
            else "NOT_RUN"
            if any(x["status"] == "NOT_RUN" for x in rows)
            else "PASS"
            if all(x["status"] == "PASS" for x in rows)
            else "UNKNOWN"
        ),
        "checks": rows,
        "defects": defects + files["defects"],
        "scientific_acceptance": "NOT_COVERED",
    }


def write_report(root: Path, manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    result = audit_run(root, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result
