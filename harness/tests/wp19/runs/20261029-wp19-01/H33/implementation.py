"""Treat a missing historical binding as legacy, not as a newly verified selection."""

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
    """Raised for an empty binding or an unregistered contract id."""


def solve(payload: dict) -> dict:
    """Return legacy_unbound when binding is null; bound only for a nonempty known list."""
    if not isinstance(payload, dict) or set(payload) != {"binding"}:
        raise InputError("schema")
    binding = payload["binding"]
    if binding is None:
        return {"status": "legacy_unbound", "verified": False}
    ids = _ids(binding)
    if not ids:
        raise InputError("empty")
    return {"status": "bound", "verified": True}


def _ids(values: object) -> list[str]:
    """Accept unique registered ids. An empty list is handled by the caller."""
    if not isinstance(values, list):
        raise InputError("binding")
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
