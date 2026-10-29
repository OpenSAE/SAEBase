from __future__ import annotations

from pathlib import Path

from src.schema import digest, require_keys, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"observed", "receipt", "files"})
    observed = {row["id"]: row for row in payload["observed"]}
    unique([row["id"] for row in payload["observed"]])
    unique([row["id"] for row in payload["receipt"]])
    if set(observed) != {row["id"] for row in payload["receipt"]}:
        raise ValueError("execution evidence")
    for row in payload["receipt"]:
        seen = observed[row["id"]]
        if row["exit"] != seen["exit"] or row["command"] != seen["command"] or seen["exit"] != 0:
            raise ValueError("false receipt")
    if any(
        "/" in row["name"] or row["name"].startswith(".") or row["sha"] != digest(row["content"])
        for row in payload["files"]
    ):
        raise ValueError("provenance")
    if any(
        "secret" in str(row).lower() or "token=" in str(row).lower() for row in payload["receipt"]
    ):
        raise ValueError("sensitive evidence")
    return {
        "verified_commands": sorted(observed),
        "source_hashes": {row["name"]: row["sha"] for row in payload["files"]},
        "failed_count": 0,
        "human_approved": False,
    }
