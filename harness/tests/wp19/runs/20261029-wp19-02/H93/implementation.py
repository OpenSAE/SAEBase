"""子树迁移: 只更新必要边"""

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
    return _edges_ok(subject.get("nodes"), subject.get("edges"))


def _unique(values: object) -> bool:
    return (
        isinstance(values, list)
        and all(isinstance(item, str) and item for item in values)
        and len(set(values)) == len(values)
    )


def _edges_ok(nodes: object, edges: object) -> bool:
    if not _unique(nodes) or not isinstance(edges, list):
        return False
    return not any(
        not isinstance(edge, list) or len(edge) != 2 or any(end not in nodes for end in edge)
        for edge in edges
    )
