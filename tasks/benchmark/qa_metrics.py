"""Reusable independent-question bootstrap metrics for evaluation tasks."""

from __future__ import annotations
import math
import random
from typing import Any

BOOTSTRAP_SEED = 20261031
BOOTSTRAP_DRAWS = 512


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left, right = int(position), min(int(position) + 1, len(ordered) - 1)
    return float(ordered[left] + (ordered[right] - ordered[left]) * (position - left))


def summarize_metric(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Core benchmark requires finite nonempty measurements")
    n = len(values)
    rnd = random.Random(seed)
    bootstrap = [
        sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "value": float(sum(values) / n),
        "n_valid": n,
        "ci_lower_95": _percentile(bootstrap, 0.025),
        "ci_upper_95": _percentile(bootstrap, 0.975),
        "bootstrap_seed": seed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "confidence_method": "iid_question_percentile_bootstrap_v1",
    }
