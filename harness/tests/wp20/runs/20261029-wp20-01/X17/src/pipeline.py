from __future__ import annotations

from pathlib import Path

from src.schema import require_keys


def execute(payload, workdir: Path):
    require_keys(payload, {"source_sha", "checks", "required", "reviewer"})
    required = payload["required"]
    if (
        len(set(required)) != len(required)
        or not required
        or not isinstance(payload["source_sha"], str)
    ):
        raise ValueError("required")
    by_id = {row["id"]: row for row in payload["checks"]}
    if len(by_id) != len(payload["checks"]) or any(name not in by_id for name in required):
        raise ValueError("missing check")
    if any(
        by_id[name]["source_sha"] != payload["source_sha"] or by_id[name]["state"] != "success"
        for name in required
    ):
        raise ValueError("incomplete release gates")
    if payload["reviewer"] != "trusted-human":
        raise ValueError("no human approval")
    return {
        "source_sha": payload["source_sha"],
        "required": sorted(required),
        "release_ready": True,
        "published": False,
    }
