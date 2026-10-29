"""Bootstrap 随机性: seed 与抽样单元可复现"""

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
    if not isinstance(subject, dict):
        return False
    done, total, coverage = subject.get("done"), subject.get("total"), subject.get("coverage")
    return (
        type(done) is int
        and type(total) is int
        and total > 0
        and 0 <= done <= total
        and _finite(coverage)
        and math.isclose(done / total, coverage, abs_tol=1e-9)
    )


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)
