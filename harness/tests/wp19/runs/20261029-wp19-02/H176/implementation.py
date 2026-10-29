"""报告数字一致: 原始输入可重算结果"""

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
    return isinstance(subject, dict) and _mae_ok(subject)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _mae_ok(subject: dict) -> bool:
    pred, target, mae = subject.get("pred"), subject.get("target"), subject.get("mae")
    return (
        isinstance(pred, list)
        and isinstance(target, list)
        and bool(pred)
        and len(pred) == len(target)
        and all(_finite(x) for x in pred + target)
        and _finite(mae)
        and math.isclose(
            sum(abs(a - b) for a, b in zip(pred, target)) / len(pred), mae, abs_tol=1e-9
        )
    )
