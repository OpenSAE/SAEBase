"""Reusable output checks for independently collected artifacts.

This module is governance-owned. Do not treat a self-reported artifact,
selected Contract, or green public fixture as an independent acceptance.
"""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath

from harness.contracts import invariants

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_ALLOWED_ARTIFACTS = {"source_reference", "relative_path", "structured_output", "bilingual_record", "tree", "numeric_output"}


def canonical_relative_path(value: str) -> str:
    """Require an unambiguous repository-relative path on every OS."""
    if not isinstance(value, str) or not value or chr(0) in value:
        raise invariants.InvariantError("Invalid relative path")
    if value.startswith(("/", "~")) or "\\" in value or _WINDOWS_DRIVE.match(value):
        raise invariants.InvariantError("Absolute or platform-specific path")
    if ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise invariants.InvariantError("Noncanonical relative path")
    if PurePosixPath(value).as_posix() != value:
        raise invariants.InvariantError("Noncanonical relative path")
    return value


def finite_json(value: object, *, max_depth: int = 24) -> None:
    """Reject non-finite *results* as well as non-finite input numbers."""
    def visit(v: object, depth: int) -> None:
        if depth > max_depth:
            raise invariants.InvariantError("Numeric structure too deep")
        if v is None or isinstance(v, (bool, int, str)):
            return
        if isinstance(v, float):
            if not math.isfinite(v):
                raise invariants.InvariantError("Non-finite numeric value")
            return
        if isinstance(v, list):
            for x in v:
                visit(x, depth + 1)
            return
        if isinstance(v, dict):
            if not all(isinstance(key, str) for key in v):
                raise invariants.InvariantError("Non-string key")
            for x in v.values():
                visit(x, depth + 1)
            return
        raise invariants.InvariantError("Unsupported JSON value")
    visit(value, 0)


def audit_artifacts(contract_ids: list[str], artifacts: list[dict], root: Path) -> list[str]:
    """Apply selected general invariants to externally inspected real outputs.

    Unknown artifact kinds are errors. This does not execute candidate code,
    verify the truth of a receipt, or grant human approval.
    """
    if not isinstance(contract_ids, list) or not all(isinstance(x, str) for x in contract_ids):
        raise invariants.InvariantError("Selected contracts must be strings")
    if not isinstance(artifacts, list):
        raise invariants.InvariantError("Missing artifact list")
    issues: list[str] = []
    for i, row in enumerate(artifacts):
        if not isinstance(row, dict) or set(row) != {"kind", "payload"}:
            issues.append(f"{i}:invalid_artifact")
            continue
        kind, value = row["kind"], row["payload"]
        if kind not in _ALLOWED_ARTIFACTS:
            issues.append(f"{i}:unsupported_artifact")
            continue
        try:
            finite_json(value)
            if kind == "source_reference" and "CT-EVIDENCE" in contract_ids:
                invariants.reference_location(value, root)
            if kind == "relative_path" and "CT-EVIDENCE" in contract_ids:
                canonical_relative_path(value)
                if not (root.resolve() / value).resolve().is_relative_to(root.resolve()):
                    raise invariants.InvariantError("Source path escapes root")
            if kind == "structured_output" and "CT-PRIVACY" in contract_ids:
                cleaned, replaced = invariants.redact_structure(value)
                if replaced or cleaned != value:
                    raise invariants.InvariantError("Unredacted sensitive content")
            if kind == "bilingual_record" and "CT-LOCALE" in contract_ids:
                if invariants.localization_defects(value):
                    raise invariants.InvariantError("Incomplete bilingual fields")
            if kind == "tree" and "CT-STRUCT" in contract_ids:
                invariants.validated_dfs(value)
        except (invariants.InvariantError, OSError, TypeError, ValueError) as exc:
            # Never embed candidate secrets or raw values in the error report.
            issues.append(f"{i}:{kind}:{type(exc).__name__}")
    return issues
