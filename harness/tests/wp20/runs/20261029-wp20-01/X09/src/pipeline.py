from __future__ import annotations

import json
from pathlib import Path

from src.schema import finite, require_keys, serial, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"activations", "steps", "mode"})
    if payload["mode"] not in ("teacher_forced", "generation"):
        raise ValueError("schema")
    values = {row["id"]: list(row["features"]) for row in payload["activations"]}
    unique(list(values))
    baseline = json.loads(serial(values))
    trace = []
    for step in payload["steps"]:
        sample, index, kind = step["sample"], step["feature"], step["op"]
        if sample not in values or type(index) is not int or not 0 <= index < len(values[sample]):
            raise ValueError("invalid target")
        current = values[sample]
        before = current[index]
        if kind == "ablate":
            current[index] = 0
        elif kind == "add":
            current[index] += step["value"]
        elif kind == "scale":
            current[index] *= step["value"]
        elif kind == "clamp":
            current[index] = min(max(current[index], step["low"]), step["high"])
        else:
            raise ValueError("operator")
        if not finite(current[index]):
            raise ValueError("nonfinite")
        trace.append(
            {
                "sample": sample,
                "feature": index,
                "op": kind,
                "before": before,
                "after": current[index],
            }
        )
    return {
        "mode": payload["mode"],
        "baseline": baseline,
        "trace": trace,
        "after": json.loads(serial(values)),
        "rollback": baseline,
    }
