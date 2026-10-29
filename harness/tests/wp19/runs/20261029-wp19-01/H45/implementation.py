"""A path-shaped string is unresolved until existence is a boolean true."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when path is empty or exists is not a real boolean."""


def solve(payload: dict) -> dict:
    """Return resolved only when exists is true. False stays unresolved."""
    if not isinstance(payload, dict) or set(payload) != {"path", "exists"}:
        raise InputError("schema")
    path = payload["path"]
    exists = payload["exists"]
    if not isinstance(path, str) or not path:
        raise InputError("path")
    if type(exists) is not bool:
        raise InputError("exists")
    return {"status": "resolved" if exists else "unresolved"}
