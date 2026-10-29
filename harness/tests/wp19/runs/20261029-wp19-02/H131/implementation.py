"""标签对齐: 重排后标签仍跟原样本"""

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
    if not isinstance(subject, dict):
        return False
    left, right = subject.get("left"), subject.get("right")
    return _unique(left) and _unique(right) and set(left) == set(right)


def _unique(values: object) -> bool:
    return (
        isinstance(values, list)
        and all(isinstance(item, str) and item for item in values)
        and len(set(values)) == len(values)
    )
