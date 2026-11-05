"""Compare claimed reads with independently observed reads. Do not invent a match."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when claimed or observed ids are duplicated or not strings."""


def solve(payload: dict) -> dict:
    """Return unverified claims and unclaimed observations, each sorted."""
    if not isinstance(payload, dict) or set(payload) != {"claimed", "observed"}:
        raise InputError("schema")
    claimed = set(_ids(payload["claimed"]))
    observed = set(_ids(payload["observed"]))
    return {
        "unverified": sorted(claimed - observed),
        "unclaimed": sorted(observed - claimed),
    }


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
