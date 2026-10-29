"""Keep the five always-on contracts even when a task omits or overrides them."""

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
    """Raised when a selected contract list is unknown, duplicated, or not a list of ids."""


def solve(payload: dict) -> dict:
    """Return always-on ids plus the selected ids, sorted and unique."""
    if not isinstance(payload, dict) or set(payload) != {"selected"}:
        raise InputError("schema")
    return {"contracts": sorted(set(ALWAYS) | set(_ids(payload["selected"])))}


def _ids(values: object) -> list[str]:
    """Accept unique registered contract ids."""
    if not isinstance(values, list):
        raise InputError("selected")
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
