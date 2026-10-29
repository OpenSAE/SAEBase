"""英语占位句: unavailable 不算有效文本"""

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
    zh, en = subject.get("zh"), subject.get("en")
    return (
        isinstance(zh, str)
        and isinstance(en, str)
        and bool(zh.strip())
        and bool(en.strip())
        and "翻译不可用" not in zh
        and "translation unavailable" not in en.lower()
    )
