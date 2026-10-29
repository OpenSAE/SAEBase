"""模块移动: 真实路径与身份迁移对应"""

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
    path = subject.get("path") if isinstance(subject, dict) else None
    return _path(path)


def _path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith(("/", "~"))
        and "\\" not in value
        and ":" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )
