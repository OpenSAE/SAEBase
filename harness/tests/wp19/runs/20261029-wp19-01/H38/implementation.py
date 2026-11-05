"""Reject governance prefixes, parent traversal, and absolute paths."""

from __future__ import annotations

BLOCKED = (".cursor/", ".github/", "harness/core/", "harness/contracts/")


class InputError(ValueError):
    """Raised when a path leaves the permitted sandbox."""


def solve(payload: dict) -> dict:
    """Return allowed relative paths in dictionary order."""
    if not isinstance(payload, dict) or set(payload) != {"paths"}:
        raise InputError("schema")
    return {"allowed": sorted(_paths(payload["paths"]))}


def _paths(values: object) -> list[str]:
    """Check each path before any implied file access."""
    if not isinstance(values, list):
        raise InputError("paths")
    seen: set[str] = set()
    clean: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise InputError("path")
        if value in seen:
            raise InputError("duplicate")
        if value.startswith("/") or ".." in value.split("/") or value.startswith(BLOCKED):
            raise InputError("governance")
        seen.add(value)
        clean.append(value)
    return clean
