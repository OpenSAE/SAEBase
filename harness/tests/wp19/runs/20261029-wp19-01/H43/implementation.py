"""Keep two distinct sources even when their values disagree."""

from __future__ import annotations


class InputError(ValueError):
    """Raised unless exactly two claims from two different nonempty sources are supplied."""


def solve(payload: dict) -> dict:
    """Return conflict or consistent, and the sorted source names."""
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        raise InputError("schema")
    claims = payload["claims"]
    if not isinstance(claims, list) or len(claims) != 2:
        raise InputError("claims")
    first = _claim(claims[0])
    second = _claim(claims[1])
    if first[0] == second[0]:
        raise InputError("sources")
    status = "conflict" if first[1] != second[1] else "consistent"
    return {"status": status, "sources": sorted([first[0], second[0]])}


def _claim(claim: object) -> tuple[str, str]:
    """Read one source name and one string value."""
    if not isinstance(claim, dict) or set(claim) != {"source", "value"}:
        raise InputError("claim")
    source = claim["source"]
    value = claim["value"]
    if not isinstance(source, str) or not source or not isinstance(value, str):
        raise InputError("sources")
    return source, value
