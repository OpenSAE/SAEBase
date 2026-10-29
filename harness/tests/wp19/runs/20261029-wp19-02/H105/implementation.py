"""循环依赖定义: 诊断，不作为证明"""

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
    return isinstance(subject, dict) and _tree_ok(subject.get("nodes"))


def _unique(values: object) -> bool:
    return (
        isinstance(values, list)
        and all(isinstance(item, str) and item for item in values)
        and len(set(values)) == len(values)
    )


def _tree_ok(rows: object) -> bool:
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows
    ):
        return False
    ids = [row["id"] for row in rows]
    if not _unique(ids):
        return False
    parents = {row["id"]: row.get("parent") for row in rows}
    for node_id in ids:
        seen: set[object] = set()
        cursor: object = node_id
        while cursor is not None:
            if cursor in seen or cursor not in parents:
                return False
            seen.add(cursor)
            cursor = parents[cursor]
    return True
