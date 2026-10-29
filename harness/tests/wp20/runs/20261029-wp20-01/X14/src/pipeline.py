from __future__ import annotations

from pathlib import Path

from src.schema import require_keys, sha


def execute(payload, workdir: Path):
    require_keys(payload, {"base", "overrides", "models", "datasets", "budget"})
    config = dict(payload["base"])
    if set(config) != {"model", "dataset", "seed"} or any(
        key not in config for key in payload["overrides"]
    ):
        raise ValueError("config")
    config.update(payload["overrides"])
    if (
        config["model"] not in payload["models"]
        or config["dataset"] not in payload["datasets"]
        or type(config["seed"]) is not int
        or type(payload["budget"]) is not int
        or not 0 <= payload["budget"] <= 4
    ):
        raise ValueError("selection")
    jobs = [
        {
            "id": f"job-{index}",
            "model": config["model"],
            "dataset": config["dataset"],
            "seed": config["seed"] + index,
        }
        for index in range(payload["budget"])
    ]
    return {
        "resolved": config,
        "config_sha256": sha(config),
        "jobs": jobs,
        "dry_run": True,
        "receipt": {"planned": len(jobs), "executed": 0},
    }
