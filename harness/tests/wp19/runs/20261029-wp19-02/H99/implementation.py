"""正反链接: term_ids 与关联节点一致"""

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
    return _reciprocal_ok(subject.get("forward"), subject.get("backward"))


def _reciprocal_ok(forward: object, backward: object) -> bool:
    if not isinstance(forward, dict) or not isinstance(backward, dict):
        return False
    left = all(
        isinstance(ids, list) and all(term in backward and node in backward[term] for term in ids)
        for node, ids in forward.items()
    )
    right = all(
        isinstance(ids, list) and all(node in forward and term in forward[node] for node in ids)
        for term, ids in backward.items()
    )
    return left and right
