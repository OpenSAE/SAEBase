"""不匹配矩阵维度: 操作前正确失败"""

from __future__ import annotations
import math


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
    return isinstance(subject, dict) and _matrix_ok(subject.get("a"), subject.get("b"))


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _matrix_ok(left: object, right: object) -> bool:
    if not isinstance(left, list) or not left or not isinstance(right, list) or not right:
        return False
    rows = left + right
    if not all(isinstance(row, list) and row and all(_finite(x) for x in row) for row in rows):
        return False
    return (
        len(set(map(len, left))) == 1
        and len(set(map(len, right))) == 1
        and len(left[0]) == len(right)
    )
