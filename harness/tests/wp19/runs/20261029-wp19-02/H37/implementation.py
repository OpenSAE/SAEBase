"""Keep the read context set identical to the required set. Extra reads fail."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when the read ids are not exactly the required ids."""


def solve(payload: dict) -> dict:
    """Return the required ids in dictionary order when the two sets match."""
    if not isinstance(payload, dict) or set(payload) != {"required", "read"}:
        raise InputError("schema")
    required = set(_ids(payload["required"]))
    read = set(_ids(payload["read"]))
    if required != read:
        raise InputError("context")
    return {"selected": sorted(required)}


def _ids(values: object) -> list[str]:
    """Accept unique nonempty strings."""
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
