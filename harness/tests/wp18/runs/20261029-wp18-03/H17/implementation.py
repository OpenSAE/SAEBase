"""Keep synthetic train, validation, and test ids disjoint."""

from __future__ import annotations

_SPLITS = ("train", "val", "test")


class InputError(ValueError):
    """Raised when a sample id is duplicated inside or across splits."""


def solve(payload: dict) -> dict:
    """Return sorted ids and counts for the three required splits."""
    if not isinstance(payload, dict) or set(payload) != {"splits"}:
        raise InputError("schema")
    splits = payload["splits"]
    if not isinstance(splits, dict) or set(splits) != set(_SPLITS):
        raise InputError("splits")
    owned: dict[str, set[str]] = {}
    ids: dict[str, list[str]] = {}
    for name in _SPLITS:
        rows = _split_ids(splits[name])
        ids[name] = rows
        owned[name] = set(rows)
    if (
        owned["train"] & owned["val"]
        or owned["train"] & owned["test"]
        or owned["val"] & owned["test"]
    ):
        raise InputError("overlap")
    return {
        "counts": {name: len(ids[name]) for name in _SPLITS},
        "ids": ids,
    }


def _split_ids(values: object) -> list[str]:
    """Sort one split after rejecting duplicate or empty ids."""
    if not isinstance(values, list):
        raise InputError("split")
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            raise InputError("id")
        seen.add(value)
    return sorted(seen)
