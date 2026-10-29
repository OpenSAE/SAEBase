"""Recall: 阈值/top-k/TP/FN 定义正确"""

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
    return isinstance(subject, dict) and _recall_ok(subject)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _recall_ok(subject: dict) -> bool:
    tp, fn, recall = subject.get("tp"), subject.get("fn"), subject.get("recall")
    return (
        type(tp) is int
        and type(fn) is int
        and tp >= 0
        and fn >= 0
        and tp + fn > 0
        and _finite(recall)
        and math.isclose(tp / (tp + fn), recall, abs_tol=1e-9)
    )
