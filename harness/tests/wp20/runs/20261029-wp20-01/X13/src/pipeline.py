from __future__ import annotations

from pathlib import Path

from src.schema import finite, require_keys


def _mean(values):
    if not values or not all(finite(value) for value in values):
        raise ValueError("invalid mean")
    return sum(values) / len(values)


def execute(payload, workdir: Path):
    require_keys(payload, {"adapters"})
    ids = [adapter["id"] for adapter in payload["adapters"]]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("duplicate or empty adapters")
    details = []
    valid = []
    for adapter in payload["adapters"]:
        rows = adapter["rows"]
        if not isinstance(rows, list) or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("sample identity")
        successful = [row for row in rows if row["status"] == "success"]
        if any(not finite(row["value"]) for row in successful):
            raise ValueError("numeric metric")
        if any(row["status"] not in ("success", "failed") for row in rows):
            raise ValueError("status")
        details.append(
            {
                "id": adapter["id"],
                "n": len(rows),
                "valid": len(successful),
                "mean": _mean([row["value"] for row in successful]) if successful else None,
            }
        )
        valid.extend(row["value"] for row in successful)
    return {
        "adapters": sorted(details, key=lambda row: row["id"]),
        "overall_mean": _mean(valid) if valid else None,
        "evaluated": len(valid),
        "total": sum(row["n"] for row in details),
    }
