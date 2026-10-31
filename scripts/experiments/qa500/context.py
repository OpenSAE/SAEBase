"""Study-specific QA500 configuration, source identities and canonical runtime paths.

Keep reusable Dataset, Model, Task, and Runtime behavior in their owning modules.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
from typing import Any
import re
from saebase.integrations.hf_assets import resolve_revision
from pathlib import Path
from omegaconf import OmegaConf
from saebase.runtime.experiment_paths import (
    artifact_path as runtime_artifact_path,
    create_layout,
    run_leaf,
    study_root,
)


# --- config: instance glue ---


@lru_cache(maxsize=None)
def _read(name: str) -> dict[str, Any]:
    if name not in {
        "protocol.json",
        "datasets.json",
        "sites.json",
        "tasks.json",
        "resources.json",
        "assets.lock.json",
    }:
        raise ValueError("Unknown QA500 configuration file")
    data = json.loads((files("conf") / "experiment" / "qa500" / name).read_text(encoding="utf-8"))
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


# --- assets: instance glue ---


def declared_revision(kind: str, name: str) -> str | None:
    key = {
        "dataset": "dataset_revisions",
        "model": "model_revisions",
        "sae": "sae_revisions",
    }.get(kind)
    if key is None:
        raise ValueError("Unknown asset type")
    value = get_assets_lock().get(key, {}).get(name)
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Malformed experiment asset lock")
    return value


def source_revision(repo_id: str, key: str) -> tuple[str, str]:
    expected = declared_revision("dataset", key)
    if expected is not None:
        return expected, "project_locked"
    return resolve_revision(repo_id, repo_type="dataset"), "first_run_frozen"


def required_publication_lock() -> bool:
    """Check full study asset coverage, never merely nonempty lock dictionaries.

    Keys are protocol dataset IDs, model registry keys, and the exact
    `model_key:sae_key:L{layer}` SAE sites. Values are full upstream commits.
    A future release must ALSO compare these revisions with downloaded assets
    and independently verified real run manifests.
    """
    lock = get_assets_lock()
    if lock.get("status") != "pinned":
        return False
    sites = get_sites()
    expected = {
        "dataset_revisions": set(get_dataset_ids()),
        "model_revisions": {entry["model_key"] for entry in sites},
        "sae_revisions": {
            f"{entry['model_key']}:{entry['sae_key']}:L{layer}"
            for entry in sites
            for layer in entry["layers"]
        },
    }
    for field, required_keys in expected.items():
        values = lock.get(field)
        if not isinstance(values, dict) or set(values) != required_keys:
            return False
        if any(
            not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None
            for sha in values.values()
        ):
            return False
    return True


# --- paths: instance glue ---


PROTOCOL_ID = get_protocol()["protocol_id"]
SEED = get_protocol()["seed"]


def _root(workspace: str | Path) -> Path:
    return Path(workspace).resolve()


def study_dir(workspace: str | Path) -> Path:
    return study_root(workspace, PROTOCOL_ID)


def inputs_dir(workspace: str | Path) -> Path:
    return study_dir(workspace) / "inputs"


def staging_dir(workspace: str | Path) -> Path:
    return study_dir(workspace) / "staging"


def receipts_dir(workspace: str | Path, model_key: str | None = None) -> Path:
    root = study_dir(workspace) / "metadata" / "receipts"
    return root / model_key if model_key else root


def task_dir(
    workspace: str | Path,
    *,
    task: str,
    dataset: str,
    model: str,
    sae_key: str | None = None,
    layer: int | None = None,
    initialize: bool = False,
) -> Path:
    """Derive one native Run leaf from the existing validated Runtime API."""
    if task not in {
        "capture",
        "benchmark",
        "intervene",
        "generate",
        "explain",
        "circuits",
        "train",
    }:
        raise ValueError("Task directory must be a registered SAEBase operation")
    site = "baseline" if sae_key is None else str(sae_key)
    segments = [task, "qa500", site]
    if layer is not None:
        if type(layer) is not int or layer < 0:
            raise ValueError("Invalid SAE layer")
        segments.append(f"L{layer}")
    cfg = OmegaConf.create(
        {
            "paths": {"root": str(_root(workspace))},
            "dataset": {"name": dataset},
            "model": {"name": model},
            "task": {"name": task, "path": "/".join(segments)},
            "experiment": {"sub_name": "qa500", "overwrite": False},
            "seed": SEED,
        }
    )
    destination = run_leaf(cfg)
    if initialize:
        destination.mkdir(parents=True, exist_ok=True)
        metadata = destination / "metadata" / "run.json"
        if not metadata.exists():
            create_layout(cfg, destination)
    return destination


def typed_file(
    run: Path,
    category: str,
    filename: str,
) -> Path:
    """Respect the existing typed-artifact contract."""
    return runtime_artifact_path(run, category, filename)
