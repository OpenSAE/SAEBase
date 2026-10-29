"""Selecting a contract never unlocks a protected governance write."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when the action is unknown or authorized is not a boolean."""


def solve(payload: dict) -> dict:
    """Allow sandbox_write. governance_write stays disallowed even if authorized is true."""
    if not isinstance(payload, dict) or set(payload) != {"action", "authorized"}:
        raise InputError("schema")
    action = payload["action"]
    authorized = payload["authorized"]
    if action not in {"sandbox_write", "governance_write"}:
        raise InputError("action")
    if type(authorized) is not bool:
        raise InputError("authorization")
    return {"allowed": action == "sandbox_write"}
