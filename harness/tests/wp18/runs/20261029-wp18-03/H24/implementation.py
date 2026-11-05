"""Resume only when the synthetic run, config, dataset, and seed all match."""

from __future__ import annotations

import re

_SHA = re.compile(r"[0-9a-f]{64}")
_FIELDS = ("run_id", "config_sha", "dataset_sha", "seed")


class InputError(ValueError):
    """Raised when a resume request does not match the saved identity."""


def solve(payload: dict) -> dict:
    """Return resume true only for an exact identity match. Nothing is merged."""
    if not isinstance(payload, dict) or set(payload) != {"saved", "request"}:
        raise InputError("schema")
    saved = _identity(payload["saved"])
    request = _identity(payload["request"])
    if saved != request:
        raise InputError("identity")
    return {"resume": True}


def _identity(value: object) -> dict[str, object]:
    """Read the four identity fields without substituting defaults."""
    if not isinstance(value, dict) or set(value) != set(_FIELDS):
        raise InputError("identity")
    run_id = value["run_id"]
    config_sha = value["config_sha"]
    dataset_sha = value["dataset_sha"]
    seed = value["seed"]
    if not isinstance(run_id, str) or not run_id:
        raise InputError("identity")
    if not isinstance(config_sha, str) or _SHA.fullmatch(config_sha) is None:
        raise InputError("identity")
    if not isinstance(dataset_sha, str) or _SHA.fullmatch(dataset_sha) is None:
        raise InputError("identity")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InputError("identity")
    return {
        "run_id": run_id,
        "config_sha": config_sha,
        "dataset_sha": dataset_sha,
        "seed": seed,
    }
