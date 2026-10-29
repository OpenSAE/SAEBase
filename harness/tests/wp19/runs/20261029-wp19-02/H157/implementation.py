"""Ablation: 只清零选中特征"""

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
    return isinstance(subject, dict) and _vector_ok(subject)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _vector_ok(subject: dict) -> bool:
    before, after, idx, mode, delta = (
        subject.get("before"),
        subject.get("after"),
        subject.get("index"),
        subject.get("mode"),
        subject.get("delta"),
    )
    if (
        not isinstance(before, list)
        or not isinstance(after, list)
        or len(before) != len(after)
        or type(idx) is not int
        or idx < 0
        or idx >= len(before)
        or not all(_finite(x) for x in before + after)
        or not _finite(delta)
    ):
        return False
    expected = list(before)
    if mode == "ablate":
        expected[idx] = 0
    elif mode == "scale":
        expected[idx] *= delta
    elif mode == "add":
        expected[idx] += delta
    elif mode == "clamp":
        expected[idx] = min(max(expected[idx], 0), delta)
    else:
        return False
    return all(math.isclose(left, right, abs_tol=1e-9) for left, right in zip(expected, after))
