"""源 SHA 变化: 旧 approval 自动失效或阻止复用"""

from __future__ import annotations
import re


class InputError(ValueError):
    """Raised when the subject is missing or violates the frozen invariant."""


def solve(payload: dict) -> dict:
    """Accept one subject that satisfies this case operator. Invalid input raises."""
    if not isinstance(payload, dict) or set(payload) != {"subject"}:
        raise InputError("schema")
    try:
        ok = _ok(payload["subject"])
    except (TypeError, KeyError, ValueError, OverflowError, ZeroDivisionError, RecursionError):
        ok = False
    if not ok:
        raise InputError("contract")
    return {"accepted": True}


def _ok(subject: object) -> bool:
    """Return true only when the subject satisfies the frozen operator."""
    if not isinstance(subject, dict):
        return False
    expected = subject.get("expected")
    return (
        isinstance(expected, str)
        and _SHA.fullmatch(expected) is not None
        and expected == subject.get("observed")
    )


_SHA = re.compile(r"[0-9a-f]{64}")
