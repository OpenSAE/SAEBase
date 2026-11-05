"""Admit a synthetic sample budget only when the request fits and GPU use is authorized."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a request exceeds budget or uses GPU without authorization."""


def solve(payload: dict) -> dict:
    """Return remaining samples and gpu_minutes. This does not probe hardware."""
    if not isinstance(payload, dict) or set(payload) != {"limit", "request", "gpu_authorized"}:
        raise InputError("schema")
    if not isinstance(payload["gpu_authorized"], bool):
        raise InputError("authorization")
    limit = _budget(payload["limit"])
    request = _budget(payload["request"])
    if request["samples"] > limit["samples"] or request["gpu_minutes"] > limit["gpu_minutes"]:
        raise InputError("budget")
    if request["gpu_minutes"] > 0 and payload["gpu_authorized"] is not True:
        raise InputError("authorization")
    return {
        "allowed": True,
        "remaining": {
            "samples": limit["samples"] - request["samples"],
            "gpu_minutes": limit["gpu_minutes"] - request["gpu_minutes"],
        },
    }


def _budget(value: object) -> dict[str, int]:
    """Require nonnegative integer samples and gpu_minutes."""
    if not isinstance(value, dict) or set(value) != {"samples", "gpu_minutes"}:
        raise InputError("budget")
    amounts: dict[str, int] = {}
    for key in ("samples", "gpu_minutes"):
        amount = value[key]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise InputError("budget")
        amounts[key] = amount
    return amounts
