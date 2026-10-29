from __future__ import annotations

import random
from pathlib import Path

from src.schema import finite


def _mean(values):
    if not values or not all(finite(value) for value in values):
        raise ValueError("invalid mean")
    return sum(values) / len(values)


def execute(payload, workdir: Path):
    if (
        set(payload) != {"groups", "seed", "draws"}
        or type(payload["seed"]) is not int
        or type(payload["draws"]) is not int
        or not 1 <= payload["draws"] <= 60
    ):
        raise ValueError("schema")
    groups = payload["groups"]
    ids = [row["id"] for row in groups]
    if (
        len(set(ids)) != len(ids)
        or not groups
        or any(
            not row["values"] or not all(finite(value) for value in row["values"]) for row in groups
        )
    ):
        raise ValueError("groups")
    ordered = sorted(groups, key=lambda row: row["id"])
    rng = random.Random(payload["seed"])
    samples = []
    stats = []
    for _ in range(payload["draws"]):
        selected = [rng.randrange(len(ordered)) for _repeat in ordered]
        samples.append([ordered[index]["id"] for index in selected])
        stats.append(_mean([value for index in selected for value in ordered[index]["values"]]))
    ranked = sorted(stats)
    return {
        "baseline": _mean([value for row in ordered for value in row["values"]]),
        "replicate_indices": samples,
        "replicate_means": stats,
        "interval": [
            ranked[int(0.025 * (len(stats) - 1))],
            ranked[int(0.975 * (len(stats) - 1))],
        ],
        "seed": payload["seed"],
    }
