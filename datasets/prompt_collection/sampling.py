"""Uniform sampling over indexed source populations; independent of split labels."""

from __future__ import annotations

import random
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor


def sample_records(total, get_row, count, seed):
    """Keep all small populations; otherwise draw uniformly without replacement.

    Empty structural text rows are not observations. Rejection is independent
    of model predictions, split labels, topic and input length.
    """
    if total < 0 or count < 1:
        raise ValueError("Invalid population or sample count")
    if total <= count:
        return [row for index in range(total) if (row := get_row(index)).get("prompt", "").strip()]
    rng = random.Random(seed)
    seen = set()
    chosen = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        while len(chosen) < count and len(seen) < total:
            indices = []
            batch_size = min(count - len(chosen), total - len(seen))
            while len(indices) < batch_size:
                index = rng.randrange(total)
                if index not in seen:
                    seen.add(index)
                    indices.append(index)
            for row in pool.map(get_row, indices):
                if row.get("prompt", "").strip():
                    chosen.append(row)
    return chosen


def union_index(parts):
    """Index every split proportionally to its actual size, without materializing rows."""
    keys = list(parts)
    bounds = []
    total = 0
    for key in keys:
        total += len(parts[key])
        bounds.append(total)

    def get(index):
        position = bisect_right(bounds, index)
        start = 0 if position == 0 else bounds[position - 1]
        return keys[position], index - start, parts[keys[position]][index - start]

    return total, get
