"""Match an expected registry version string to an observed receipt version."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when versions are missing, non-string, or disagree."""


def solve(payload: dict) -> dict:
    """Return matched true only when both nonempty strings are identical."""
    if not isinstance(payload, dict) or set(payload) != {"expected", "observed"}:
        raise InputError("schema")
    expected = payload["expected"]
    observed = payload["observed"]
    if not isinstance(expected, str) or not expected:
        raise InputError("version")
    if not isinstance(observed, str) or not observed:
        raise InputError("version")
    if expected != observed:
        raise InputError("stale")
    return {"matched": True, "version": expected}
