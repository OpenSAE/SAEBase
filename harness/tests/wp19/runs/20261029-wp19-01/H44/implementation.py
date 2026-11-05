"""Prefer the actual source over a comment when the two disagree."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when comment or actual is missing or is not a string."""


def solve(payload: dict) -> dict:
    """Return the actual text as authoritative and whether it disagrees with the comment."""
    if not isinstance(payload, dict) or set(payload) != {"comment", "actual"}:
        raise InputError("schema")
    comment = payload["comment"]
    actual = payload["actual"]
    if not isinstance(comment, str) or not isinstance(actual, str):
        raise InputError("source")
    return {"authoritative": actual, "disagreement": comment != actual}
