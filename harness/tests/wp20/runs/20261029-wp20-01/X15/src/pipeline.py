from __future__ import annotations

import json
from pathlib import Path

from src.schema import require_keys, serial, sha


def execute(payload, workdir: Path):
    require_keys(payload, {"identity", "stages", "interrupt_after", "resume_identity"})
    identity = payload["identity"]
    if (
        not isinstance(identity, dict)
        or set(identity) != {"run", "seed", "config_sha", "data_sha"}
        or identity != payload["resume_identity"]
        or type(identity["seed"]) is not int
    ):
        raise ValueError("checkpoint mismatch")
    stage_ids = [stage["id"] for stage in payload["stages"]]
    if (
        len(set(stage_ids)) != len(stage_ids)
        or type(payload["interrupt_after"]) is not int
        or not 0 <= payload["interrupt_after"] <= len(stage_ids)
    ):
        raise ValueError("stages")
    path = Path(workdir) / "checkpoint.json"
    stages = [{"id": stage["id"], "value": stage["value"]} for stage in payload["stages"]]
    partial = stages[: payload["interrupt_after"]]
    path.write_text(serial({"identity": identity, "completed": partial}), encoding="utf-8")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    remaining = stages[len(checkpoint["completed"]) :]
    completed = checkpoint["completed"] + remaining
    path.write_text(serial({"identity": identity, "completed": completed}), encoding="utf-8")
    repeat = json.loads(path.read_text(encoding="utf-8"))
    if repeat["completed"] != stages:
        raise ValueError("non-idempotent")
    return {
        "checkpoint_sha256": sha({"identity": identity, "completed": partial}),
        "interrupted_at": payload["interrupt_after"],
        "completed": completed,
        "resumed": len(remaining),
        "idempotent": repeat["completed"] == stages,
    }
