"""Evidence-based PASS/FAIL/BLOCKED/NOT_RUN decisions for Precheck."""

from __future__ import annotations

from typing import Any

STATES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_RUN"})


def blocked(
    *,
    stage: str,
    reason: str,
    code: str,
    attempted: list[str] | None = None,
    repair: str,
    retryable: bool = True,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "stage": stage,
        "code": code,
        "reason": reason,
        "attempted": attempted or [],
        "repair": repair,
        "retryable": retryable,
        "evidence": evidence or {},
    }


def error_result(exc: Exception, *, stage: str, attempted: list[str]) -> dict[str, Any]:
    """Do not turn arbitrary runtime errors into environmental BLOCKED states."""
    name = type(exc).__name__
    reason = f"{name}: {str(exc)[:900]}"
    msg = str(exc).lower()
    if isinstance(exc, PermissionError):
        code, repair = (
            "BLOCKED_AUTH",
            "Approve the specific operation and its GPU/network/external-service budget.",
        )
    elif isinstance(exc, (ModuleNotFoundError, ImportError)):
        code, repair = (
            "BLOCKED_DEPENDENCY",
            "Run Setup with explicitly authorized package installation, then retry.",
        )
    elif isinstance(exc, (FileNotFoundError, IsADirectoryError)):
        code, repair = (
            "BLOCKED_ASSET",
            "Prepare the missing model, dataset, checkpoint, or configured input file.",
        )
    elif "out of memory" in msg or "cuda unavailable" in msg or "no cuda" in msg:
        code, repair = (
            "BLOCKED_RESOURCE",
            "Choose a machine with enough available resources; do not substitute a different scientific configuration.",
        )
    elif (
        name == "UnsupportedCapability"
        or "not compatible" in msg
        or "unsupported" in msg
        or "sole registered" in msg
    ):
        code, repair = (
            "BLOCKED_COMPATIBILITY",
            "Choose an actually compatible model, SAE site, backend or Task; no silent substitution.",
        )
    elif isinstance(exc, ValueError) and any(
        word in msg
        for word in (
            "missing",
            "provide",
            "require",
            "needs",
            "empty",
            "no default",
            "not registered",
            "not in model",
        )
    ):
        code, repair = (
            "BLOCKED_CONFIGURATION",
            "Supply the missing explicit configuration, resource, or input and retry this check.",
        )
    else:
        return {
            "status": "FAIL",
            "stage": stage,
            "code": "EXECUTION_ERROR",
            "reason": reason,
            "attempted": attempted,
            "repair": "Inspect this Task's error and its declared input/output contract before retrying.",
            "retryable": False,
            "evidence": {},
        }
    return blocked(stage=stage, code=code, reason=reason, attempted=attempted, repair=repair)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        state = "NOT_RUN"
    elif any(row.get("status") == "FAIL" for row in rows):
        state = "FAIL"
    elif any(row.get("status") == "BLOCKED" for row in rows):
        state = "BLOCKED"
    elif any(row.get("status") == "NOT_RUN" for row in rows):
        state = "NOT_RUN"
    elif all(row.get("status") == "PASS" for row in rows):
        state = "PASS"
    else:
        state = "FAIL"
    return {
        "status": state,
        "counts": {
            s: sum(x.get("status") == s for x in rows)
            for s in ("PASS", "FAIL", "BLOCKED", "NOT_RUN")
        },
        "checked": len(rows),
        "scientific_acceptance": "NOT_COVERED",
        "independent_harness_acceptance": "NOT_COVERED",
    }
