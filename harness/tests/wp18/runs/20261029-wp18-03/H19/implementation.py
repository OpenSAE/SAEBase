"""Compute MAE for two equal-length synthetic numeric vectors."""

from __future__ import annotations


class InputError(ValueError):
    """Raised for empty, mismatched, non-finite, or boolean-as-number vectors."""


def solve(payload: dict) -> dict:
    """Return the population size and mean absolute error. No row is dropped."""
    if not isinstance(payload, dict) or set(payload) != {"pred", "target"}:
        raise InputError("schema")
    pred = _numbers(payload["pred"])
    target = _numbers(payload["target"])
    if not pred or len(pred) != len(target):
        raise InputError("length")
    total = sum(abs(left - right) for left, right in zip(pred, target, strict=True))
    count = len(pred)
    if all(isinstance(item, int) for item in pred + target) and total % count == 0:
        mae: int | float = total // count
    else:
        mae = total / count
    return {"n": count, "mae": mae}


def _numbers(values: object) -> list[int | float]:
    """Accept finite ints and floats. Booleans are not numbers here."""
    if not isinstance(values, list):
        raise InputError("vector")
    numbers: list[int | float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError("type")
        if (
            isinstance(value, float)
            and value != value
            or isinstance(value, float)
            and value
            in {
                float("inf"),
                float("-inf"),
            }
        ):
            raise InputError("type")
        numbers.append(value)
    return numbers
