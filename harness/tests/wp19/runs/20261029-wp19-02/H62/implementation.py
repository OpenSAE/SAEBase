"""异常日志泄密: stderr 与返回值同级脱敏"""

from __future__ import annotations
import re


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
    return _walk_redacted(subject.get("record"))


def _walk_redacted(value: object, key: str = "", depth: int = 0) -> bool:
    if depth > 24:
        return False
    if isinstance(value, dict):
        return all(
            isinstance(name, str) and _walk_redacted(item, name, depth + 1)
            for name, item in value.items()
        )
    if isinstance(value, list):
        return all(_walk_redacted(item, key, depth + 1) for item in value)
    if isinstance(value, str):
        if re.search(r"(?i)token|secret|cookie|password|private.?key|credential", key):
            return value == "[REDACTED]"
        return re.search(r"(?i)(token|secret|password)\s*[:=]\s*\S+", value) is None
    return value is None or type(value) in (bool, int, float)
