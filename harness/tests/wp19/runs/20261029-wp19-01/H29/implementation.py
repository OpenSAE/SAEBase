"""Accept only contract ids that already exist in the frozen catalog."""

from __future__ import annotations

KNOWN = {
    "CT-AST",
    "CT-BASE",
    "CT-CHECK",
    "CT-CONFIG",
    "CT-CONTEXT",
    "CT-DATA",
    "CT-DETERMINISM",
    "CT-ENV",
    "CT-ERROR",
    "CT-EVIDENCE",
    "CT-INTERFACE",
    "CT-LOCALE",
    "CT-NAMESPACE",
    "CT-PRIVACY",
    "CT-RECOVERY",
    "CT-RELEASE",
    "CT-RESOURCE",
    "CT-REVIEW",
    "CT-SCOPE",
    "CT-STRUCT",
    "CT-TRACE",
    "CT-TRUTH",
}


class InputError(ValueError):
    """Raised for an unregistered or duplicated contract id."""


def solve(payload: dict) -> dict:
    """Return registered ids in dictionary order."""
    if not isinstance(payload, dict) or set(payload) != {"ids"}:
        raise InputError("schema")
    return {"ids": sorted(_ids(payload["ids"]))}


def _ids(values: object) -> list[str]:
    """Reject unknown and duplicate catalog ids."""
    if not isinstance(values, list):
        raise InputError("ids")
    seen: set[str] = set()
    clean: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value not in KNOWN:
            raise InputError("id")
        if value in seen:
            raise InputError("duplicate")
        seen.add(value)
        clean.append(value)
    return clean
