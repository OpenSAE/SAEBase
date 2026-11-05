"""Normalize an explicit list of repository-relative paths."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a path list is not a safe relative inventory."""


def solve(payload: dict) -> dict:
    """Return the same paths in dictionary order, without reading them."""
    if not isinstance(payload, dict) or set(payload) != {"paths"}:
        raise InputError("schema")
    return {"paths": _paths(payload["paths"])}


def _paths(values: object) -> list[str]:
    """Reject absolute paths, traversal, empty segments, duplicates, and backslashes."""
    if not isinstance(values, list):
        raise InputError("paths")
    accepted: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or "\\" in value or value.startswith("/"):
            raise InputError("path")
        parts = value.split("/")
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise InputError("path")
        if value in seen:
            raise InputError("duplicate")
        seen.add(value)
        accepted.append(value)
    return sorted(accepted)
