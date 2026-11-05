"""Switch one SAEBase Atlas view without writing over the other views."""

from __future__ import annotations

_VIEWS = {"core", "knowledge", "code", "environment", "context"}


class InputError(ValueError):
    """Raised for a foreign project or a view that is not present in this state."""


def solve(payload: dict) -> dict:
    """Set the active view to next and return the original per-view focus values."""
    if not isinstance(payload, dict) or set(payload) != {"project", "views", "active", "next"}:
        raise InputError("schema")
    if payload["project"] != "SAEBase":
        raise InputError("project")
    views = _views(payload["views"])
    active = payload["active"]
    nxt = payload["next"]
    if active not in views or nxt not in views:
        raise InputError("view")
    return {"active": nxt, "views": views}


def _views(value: object) -> dict[str, dict[str, str]]:
    """Copy focus for the views that are both known and present."""
    if not isinstance(value, dict) or not value:
        raise InputError("views")
    copied: dict[str, dict[str, str]] = {}
    for name, state in value.items():
        if name not in _VIEWS or not isinstance(state, dict) or set(state) != {"focus"}:
            raise InputError("view")
        focus = state["focus"]
        if not isinstance(focus, str) or not focus:
            raise InputError("focus")
        copied[name] = {"focus": focus}
    return copied
