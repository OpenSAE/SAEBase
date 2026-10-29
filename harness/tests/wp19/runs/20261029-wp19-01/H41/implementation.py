"""Split context references into known and unknown ids. Unknown ids stay unresolved."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a known or referenced id is empty, non-string, or duplicated."""


def solve(payload: dict) -> dict:
    """Return resolved ids that exist in known, and unresolved ids that do not."""
    if not isinstance(payload, dict) or set(payload) != {"known", "references"}:
        raise InputError("schema")
    known = set(_ids(payload["known"]))
    refs = set(_ids(payload["references"]))
    return {
        "resolved": sorted(refs & known),
        "unresolved": sorted(refs - known),
    }


def _ids(values: object) -> list[str]:
    """Accept unique nonempty strings without inventing replacements."""
    if not isinstance(values, list):
        raise InputError("ids")
    seen: set[str] = set()
    clean: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise InputError("id")
        if value in seen:
            raise InputError("duplicate")
        seen.add(value)
        clean.append(value)
    return clean
