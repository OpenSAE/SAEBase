"""Compare synthetic AST ids and exact signatures."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when an AST side contains a duplicate or a non-string signature."""


def solve(payload: dict) -> dict:
    """Return sorted added, removed, and signature-changed ids."""
    if not isinstance(payload, dict) or set(payload) != {"before", "after"}:
        raise InputError("schema")
    before = _side(payload["before"])
    after = _side(payload["after"])
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        node_id for node_id in before.keys() & after.keys() if before[node_id] != after[node_id]
    )
    return {"added": added, "removed": removed, "changed": changed}


def _side(values: object) -> dict[str, str]:
    """Index one side by id. Signature comparison is exact and order-independent."""
    if not isinstance(values, list):
        raise InputError("side")
    rows: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"id", "signature"}:
            raise InputError("node")
        node_id = value["id"]
        signature = value["signature"]
        if not isinstance(node_id, str) or not node_id or node_id in rows:
            raise InputError("id")
        if not isinstance(signature, str):
            raise InputError("signature")
        rows[node_id] = signature
    return rows
