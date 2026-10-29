"""重复结果 ID: 内容冲突不覆盖"""

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
    return isinstance(subject, dict) and _unique(subject.get("ids"))


def _unique(values: object) -> bool:
    return (
        isinstance(values, list)
        and all(isinstance(item, str) and item for item in values)
        and len(set(values)) == len(values)
    )
