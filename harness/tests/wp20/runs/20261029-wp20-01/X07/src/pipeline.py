from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.schema import finite, require_keys, serial, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"model", "sae", "tokens", "batch_size"})
    model, sae = payload["model"], payload["sae"]
    if (
        model["id"] != sae["model_id"]
        or model["width"] != sae["d_in"]
        or sae["layer"] not in model["layers"]
        or payload["batch_size"] < 1
    ):
        raise ValueError("SAE identity")
    unique([row["id"] for row in payload["tokens"]])
    if any(
        len(row["activation"]) != model["width"]
        or not all(finite(value) for value in row["activation"])
        for row in payload["tokens"]
    ):
        raise ValueError("shape")
    ordered = sorted(payload["tokens"], key=lambda row: row["id"])
    cache = {row["id"]: row["activation"] for row in ordered}
    identity = {
        "model": model["id"],
        "checkpoint": model["sha"],
        "sae": sae["id"],
        "layer": sae["layer"],
        "tokenizer": model["tokenizer"],
    }
    path = Path(workdir) / "activation-cache.json"
    path.write_text(serial({"identity": identity, "cache": cache}), encoding="utf-8")
    replay = json.loads(path.read_text(encoding="utf-8"))
    if replay["identity"] != identity or replay["cache"] != cache:
        raise ValueError("cache corrupt")
    return {
        "identity": identity,
        "samples": sorted(cache),
        "cache": cache,
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
