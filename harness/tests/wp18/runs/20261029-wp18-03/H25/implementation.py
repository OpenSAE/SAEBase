"""Normalize a synthetic batch by id. Duplicates reject the whole batch."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when an item id is empty or repeated. No partial batch is returned."""


def solve(payload: dict) -> dict:
    """Return items sorted by id. Input order does not change the result."""
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise InputError("schema")
    items = payload["items"]
    if not isinstance(items, list):
        raise InputError("items")
    accepted: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        row = _item(item)
        item_id = row["id"]
        if not isinstance(item_id, str) or item_id in seen:
            raise InputError("duplicate")
        seen.add(item_id)
        accepted.append(row)
    accepted.sort(key=lambda row: str(row["id"]))
    return {"items": accepted}


def _item(value: object) -> dict[str, object]:
    """Keep one nonempty id and its original scalar value."""
    if not isinstance(value, dict) or set(value) != {"id", "value"}:
        raise InputError("item")
    item_id = value["id"]
    item_value = value["value"]
    if not isinstance(item_id, str) or not item_id:
        raise InputError("id")
    if (
        isinstance(item_value, bool)
        or not isinstance(item_value, (int, float, str))
        and item_value is not None
    ):
        raise InputError("value")
    if isinstance(item_value, float) and (
        item_value != item_value or item_value in {float("inf"), float("-inf")}
    ):
        raise InputError("value")
    return {"id": item_id, "value": item_value}
