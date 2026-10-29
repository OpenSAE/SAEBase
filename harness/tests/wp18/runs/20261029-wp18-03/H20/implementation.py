"""Canonicalize one explicit model, dataset, and seed configuration."""

from __future__ import annotations

import json


class InputError(ValueError):
    """Raised when a config has unknown fields or a non-integer seed."""


def solve(payload: dict) -> dict:
    """Return compact sorted JSON. Key order in the input does not change it."""
    if not isinstance(payload, dict) or set(payload) != {"config"}:
        raise InputError("schema")
    return {"canonical": _config(payload["config"])}


def _config(value: object) -> str:
    """Require exactly model, dataset, and an integer seed."""
    if not isinstance(value, dict) or set(value) != {"model", "dataset", "seed"}:
        raise InputError("config")
    model = value["model"]
    dataset = value["dataset"]
    seed = value["seed"]
    if not isinstance(model, str) or not model or not isinstance(dataset, str) or not dataset:
        raise InputError("config")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InputError("seed")
    body = {"dataset": dataset, "model": model, "seed": seed}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
