"""失败记录丢失: 新一轮不得覆盖旧失败"""

from __future__ import annotations


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
    return (
        isinstance(subject, dict)
        and "before" in subject
        and "after" in subject
        and type(subject["before"]) is type(subject["after"])
        and subject["before"] == subject["after"]
    )
