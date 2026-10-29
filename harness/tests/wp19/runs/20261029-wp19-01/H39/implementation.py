"""A path that exists is still unverified unless it is in the allowed set."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a reference is outside the allowed list or an id is duplicated."""


def solve(payload: dict) -> dict:
    """Return verified refs that all belong to the allowed set, preserving input order after sort."""
    if not isinstance(payload, dict) or set(payload) != {"allowed", "refs"}:
        raise InputError("schema")
    allowed = set(_ids(payload["allowed"]))
    refs = _ids(payload["refs"])
    if not set(refs) <= allowed:
        raise InputError("unauthorized")
    return {"verified": sorted(refs)}


def _ids(values: object) -> list[str]:
    """Accept unique nonempty strings."""
    if not isinstance(values, list):
        raise InputError("ids")
    seen: set[str] = set()
    clean: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise InputError("id")
        if value in seen:
            raise InputError("duplicate")
        seen.add(value)
        clean.append(value)
    return clean
