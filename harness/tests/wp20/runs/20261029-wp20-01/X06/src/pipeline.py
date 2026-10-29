from __future__ import annotations

from pathlib import Path

from src.schema import finite, require_keys, sha, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"records", "selected"})
    if not isinstance(payload["records"], list):
        raise ValueError("schema")
    records = payload["records"]
    unique([row["id"] for row in records])
    if not all(
        row["split"] in ("train", "val", "test")
        and type(row["label"]) is int
        and finite(row["score"])
        for row in records
    ):
        raise ValueError("record")
    indexed = {row["id"]: row for row in records}
    unique(payload["selected"])
    if not set(payload["selected"]) <= set(indexed):
        raise ValueError("unknown sample")
    splits = {
        name: sorted(row["id"] for row in records if row["split"] == name)
        for name in ("train", "val", "test")
    }
    chosen = [indexed[key] for key in sorted(payload["selected"])]
    predictions = [
        {"id": row["id"], "score": row["score"], "label": row["label"]} for row in chosen
    ]
    return {
        "manifest": {
            "sha256": sha(sorted(records, key=lambda row: row["id"])),
            "count": len(records),
        },
        "splits": splits,
        "predictions": predictions,
        "coverage": {
            "evaluated": len(chosen),
            "total": len(records),
            "fraction": len(chosen) / len(records) if records else 0,
        },
    }
