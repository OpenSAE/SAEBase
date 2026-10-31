"""Project-neutral SAEBase Runtime paths for the QA500 Experiment.

Task results use the official `experiment/<dataset>/<model>/<task.path>/<run>`
layout. A study-level directory stores only frozen input identities, cross-run
receipts and staging. This module owns NO new experiment artifact contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from saebase.runtime.experiment_paths import (
    artifact_path as runtime_artifact_path,
    create_layout,
    run_leaf,
    study_root,
)
from experiments.qa500.config import get_protocol


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
        "capture", "benchmark", "intervene", "generate",
        "explain", "circuits", "train",
    }:
        raise ValueError("Task directory must be a registered SAEBase operation")
    site = "baseline" if sae_key is None else str(sae_key)
    segments = [task, "qa500", site]
    if layer is not None:
        if type(layer) is not int or layer < 0:
            raise ValueError("Invalid SAE layer")
        segments.append(f"L{layer}")
    cfg = OmegaConf.create({
        "paths": {"root": str(_root(workspace))},
        "dataset": {"name": dataset},
        "model": {"name": model},
        "task": {"name": task, "path": "/".join(segments)},
        "experiment": {"sub_name": "qa500", "overwrite": False},
        "seed": SEED,
    })
    destination = run_leaf(cfg)
    if initialize:
        destination.mkdir(parents=True, exist_ok=True)
        metadata = destination / "metadata" / "run.json"
        if not metadata.exists():
            create_layout(cfg, destination)
    return destination


def typed_file(
    run: Path, category: str, filename: str,
) -> Path:
    """Respect the existing typed-artifact contract."""
    return runtime_artifact_path(run, category, filename)
