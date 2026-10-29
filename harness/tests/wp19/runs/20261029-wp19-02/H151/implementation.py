"""TopK 并列值: 固定 tie-break"""

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
    return isinstance(subject, dict) and _topk_ok(subject)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _topk_ok(subject: dict) -> bool:
    scores, k, chosen = subject.get("scores"), subject.get("k"), subject.get("chosen")
    return (
        isinstance(scores, list)
        and all(_finite(x) for x in scores)
        and type(k) is int
        and 0 <= k <= len(scores)
        and isinstance(chosen, list)
        and chosen == sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    )
