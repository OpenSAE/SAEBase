"""Refuse a local minimum that is weaker than the global contract floor."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a local requirement would weaken a global integer floor."""


def solve(payload: dict) -> dict:
    """Return the stricter integer minimum. Booleans are not integers here."""
    if not isinstance(payload, dict) or set(payload) != {"global_min", "local_min"}:
        raise InputError("schema")
    global_min = payload["global_min"]
    local_min = payload["local_min"]
    if type(global_min) is not int or type(local_min) is not int:
        raise InputError("type")
    if global_min < 0 or local_min < global_min:
        raise InputError("weaken")
    return {"minimum": local_min}
