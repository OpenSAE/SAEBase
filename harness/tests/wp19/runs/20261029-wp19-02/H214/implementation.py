"""依赖锁定: wheel 与安装依赖一致"""

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
    return isinstance(subject, dict) and _manifest_ok(subject.get("files"))


_SHA = re.compile(r"[0-9a-f]{64}")


def _manifest_ok(files: object) -> bool:
    if not isinstance(files, list):
        return False
    ids: list[str] = []
    for row in files:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("sha"), str)
            or _SHA.fullmatch(row["sha"]) is None
        ):
            return False
        ids.append(row["id"])
    return len(ids) == len(set(ids))
