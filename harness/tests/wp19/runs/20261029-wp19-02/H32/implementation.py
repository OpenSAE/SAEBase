"""Union several task groups with the always-on contracts, without duplicates."""

from __future__ import annotations

ALWAYS = ("CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE")
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
    """Raised when a group is not a unique list of registered contract ids."""


def solve(payload: dict) -> dict:
    """Merge every group onto the always-on set and return a sorted union."""
    if not isinstance(payload, dict) or set(payload) != {"groups"}:
        raise InputError("schema")
    groups = payload["groups"]
    if not isinstance(groups, list):
        raise InputError("groups")
    merged = set(ALWAYS)
    for group in groups:
        if not isinstance(group, list):
            raise InputError("group")
        merged.update(_ids(group))
    return {"contracts": sorted(merged)}


def _ids(values: list[object]) -> list[str]:
    """Reject unknown and duplicate ids inside one group."""
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
