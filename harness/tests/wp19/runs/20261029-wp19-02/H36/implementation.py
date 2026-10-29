"""Accept a catalog upgrade only when always-on ids remain and nothing is dropped."""

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
    """Raised when a new catalog drops an old id or loses an always-on contract."""


def solve(payload: dict) -> dict:
    """Return compatible true only when old <= new and always-on ids stay present."""
    if not isinstance(payload, dict) or set(payload) != {"old", "new"}:
        raise InputError("schema")
    old = set(_ids(payload["old"]))
    new = set(_ids(payload["new"]))
    if not set(ALWAYS) <= new or not old <= new:
        raise InputError("weakening")
    return {"compatible": True}


def _ids(values: object) -> list[str]:
    """Accept unique registered contract ids."""
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
