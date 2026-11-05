"""Check reciprocal node and glossary links without repairing them."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when an id is duplicated or a link has no reciprocal entry."""


def solve(payload: dict) -> dict:
    """Return node:term pairs that are present in both directions."""
    if not isinstance(payload, dict) or set(payload) != {"nodes", "terms"}:
        raise InputError("schema")
    nodes = _rows(payload["nodes"], "term_ids")
    terms = _rows(payload["terms"], "node_ids")
    linked: list[str] = []
    for node_id, term_ids in nodes.items():
        for term_id in term_ids:
            if term_id not in terms or node_id not in terms[term_id]:
                raise InputError("reciprocal")
            linked.append(f"{node_id}:{term_id}")
    for term_id, node_ids in terms.items():
        for node_id in node_ids:
            if node_id not in nodes or term_id not in nodes[node_id]:
                raise InputError("reciprocal")
    return {"linked": linked}


def _rows(values: object, field: str) -> dict[str, list[str]]:
    """Index unique row ids to their unique reference ids."""
    if not isinstance(values, list):
        raise InputError(field)
    rows: dict[str, list[str]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"id", field}:
            raise InputError(field)
        row_id = value["id"]
        refs = value[field]
        if not isinstance(row_id, str) or not row_id or row_id in rows:
            raise InputError("id")
        if not isinstance(refs, list):
            raise InputError(field)
        seen: set[str] = set()
        clean: list[str] = []
        for ref in refs:
            if not isinstance(ref, str) or not ref or ref in seen:
                raise InputError(field)
            seen.add(ref)
            clean.append(ref)
        rows[row_id] = clean
    return rows
