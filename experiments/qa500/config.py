"""Experiment definition loader. Component implementations live elsewhere.

The JSON files under conf/experiment/qa500 are the only scientific
configuration source; this module merely validates and resolves them.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
from typing import Any


@lru_cache(maxsize=None)
def _read(name: str) -> dict[str, Any]:
    if name not in {
        "protocol.json", "datasets.json", "sites.json", "tasks.json",
        "resources.json", "assets.lock.json",
    }:
        raise ValueError("Unknown QA500 configuration file")
    data = json.loads(
        (files("conf") / "experiment" / "qa500" / name).read_text(encoding="utf-8")
    )
    if not isinstance(data, dict):
        raise TypeError("QA500 configuration must be a JSON object")
    return data


def get_protocol() -> dict[str, Any]:
    value = _read("protocol.json")
    if value.get("schema") != "saebase.experiment.protocol.v1":
        raise ValueError("Unknown experiment protocol schema")
    return value


def get_dataset_ids() -> tuple[str, ...]:
    value = _read("datasets.json")
    items = value["dataset_ids"]
    if not isinstance(items, list) or not items or len(items) != len(set(items)):
        raise ValueError("Dataset selection must be nonempty, unique and ordered")
    return tuple(items)


def get_sites() -> list[dict[str, Any]]:
    items = _read("sites.json")["models"]
    if not isinstance(items, list) or len({x["model_key"] for x in items}) != len(items):
        raise ValueError("QA500 model selection must have unique model identities")
    return items


def get_tasks() -> dict[str, Any]:
    return _read("tasks.json")


def get_resources() -> dict[str, Any]:
    return _read("resources.json")


def get_assets_lock() -> dict[str, Any]:
    return _read("assets.lock.json")
