"""Task-hierarchical experiment artifact contract used by every DLBase task.

Path: experiment/<dataset>/<model>/<task>/<subtask...>/<sub_name>_<seed>/
Task-level temporary logs: experiment/<dataset>/<model>/<task>/<subtask...>/_tmp/<session>/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
import datetime as dt

PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
ARTIFACT_DIRS = {
    "checkpoints": "Model checkpoints (.ckpt, .pth)",
    "metrics": "Machine-readable metric records (.json, .jsonl)",
    "tables": "Tabular evaluations (.csv, .parquet, .tsv)",
    "figures": "Plots, visualizations (.png, .pdf, .svg)",
    "logs": "Training and TensorBoard logs (.log, events.out.tfevents.*)",
    "metadata": "Run configuration, provenance and manifest (.json, .yaml)",
    "reports": "Human-readable run summary and analysis (.md, .log, .txt)",
    "tmp": "Ephemeral intermediate files scoped to this exact run",
}


def _component(value: Any, what: str) -> str:
    text = str(value)
    if not PART.fullmatch(text):
        raise ValueError(f"Unsafe {what}: {text!r} (letters, digits, - and _ only)")
    return text


def task_components(cfg) -> list[str]:
    raw = str(cfg.task.get("path", cfg.task.name))
    parts = raw.split("/")
    if not parts or any(not PART.fullmatch(p) for p in parts):
        raise ValueError(f"Invalid task.path: {raw!r}")
    name = _component(cfg.task.name, "task.name")
    if parts[0] != name:
        raise ValueError(f"task.path must start with task.name ({name!r}): {raw!r}")
    if len(parts) > 8:
        raise ValueError("task.path exceeds the supported maximum depth of 8")
    return parts


def task_root(cfg) -> Path:
    base = Path(str(cfg.paths.root)).resolve() / "experiment"
    return (
        base
        / _component(cfg.dataset.name, "dataset.name")
        / _component(cfg.model.name, "model.name")
        / Path(*task_components(cfg))
    )


def run_leaf(cfg) -> Path:
    sub = _component(cfg.experiment.sub_name, "experiment.sub_name")
    seed = _component(cfg.seed, "seed")
    return task_root(cfg) / f"{sub}_{seed}"


def check_run_location(cfg, cwd: str | Path) -> Path:
    expected = run_leaf(cfg).resolve(strict=False)
    actual = Path(cwd).resolve(strict=False)
    if expected == actual:
        return actual
    # Hydra multirun jobs use _sweeps/<timestamp>/<hydra.job.num>. Accept this
    # alternate layout only when Hydra confirms a MULTIRUN, never by path alone.
    try:
        from hydra.core.hydra_config import HydraConfig

        mode = str(HydraConfig.get().mode).upper()
    except (ImportError, ValueError, AttributeError):
        mode = ""
    sweep_root = (task_root(cfg) / "_sweeps").resolve(strict=False)
    if "MULTIRUN" in mode:
        try:
            rel = actual.relative_to(sweep_root)
        except ValueError:
            pass
        else:
            if len(rel.parts) == 2 and all(PART.fullmatch(part) for part in rel.parts):
                return actual
    raise ValueError(f"Hydra output location mismatch: expected {expected}, got {actual}")


def guard_existing_run(cfg, cwd: str | Path) -> None:
    """Do not silently delete existing reproducible artifacts in a reused run leaf."""
    folder = check_run_location(cfg, cwd)
    if not folder.exists():
        return
    prior = [p for p in folder.iterdir() if p.name != ".hydra"]
    if (
        prior
        and bool(cfg.experiment.get("overwrite", False))
        and not bool(cfg.train.get("facets", {}).get("clear_experiment_leaf", True))
    ):
        raise ValueError(
            "Overwriting an existing run requires train.facets.clear_experiment_leaf=true"
        )
    if prior and not bool(cfg.experiment.get("overwrite", False)):
        shown = ", ".join(p.name for p in prior[:5])
        raise FileExistsError(
            f"Experiment leaf contains artifacts ({shown}); pick a new sub_name/seed, "
            "or explicitly set experiment.overwrite=true to replace them."
        )


def prepare_run(cfg, cwd: str | Path, *, note: str = "") -> dict:
    """Apply the reproducible artifact contract for every Task at DLBase entry.

    Reused run leaves fail closed by default. Even for explicit overwrites,
    cleanup is limited to this one validated leaf, preserving Hydra .hydra/.
    """
    folder = check_run_location(cfg, cwd)
    guard_existing_run(cfg, folder)
    facets = cfg.train.get("facets", {})
    if bool(facets.get("clear_experiment_leaf", True)):
        for item in folder.iterdir():
            if item.name == ".hydra":
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    return create_layout(cfg, folder, note=note)


def create_layout(cfg, cwd: str | Path, *, note: str = "") -> dict:
    folder = check_run_location(cfg, cwd)
    for sub in ARTIFACT_DIRS:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    (folder / "analysis").mkdir(parents=True, exist_ok=True)
    try:
        from omegaconf import OmegaConf

        cfg_text = OmegaConf.to_yaml(cfg, resolve=True)
    except ImportError:
        cfg_text = str(cfg)
    metadata = {
        "schema_version": 1,
        "dataset": str(cfg.dataset.name),
        "model": str(cfg.model.name),
        "task": str(cfg.task.name),
        "task_hierarchy": task_components(cfg),
        "sub_name": str(cfg.experiment.sub_name),
        "seed": str(cfg.seed),
        "run_leaf": folder.name,
        "config_sha256": hashlib.sha256(cfg_text.encode("utf-8")).hexdigest(),
        "note": note
        or "Run metadata generated by DLBase; experiment outputs must follow this schema",
        "artifacts": ARTIFACT_DIRS,
        "analysis_groups": [],
        "evaluation_caveat": "Check the task-specific dataset split note; validation may not be independent of test.",
    }
    dest = folder / "metadata" / "run.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return metadata


def artifact_path(cwd: str | Path, category: str, name: str | None = None) -> Path:
    if category not in ARTIFACT_DIRS:
        raise ValueError(f"Unsupported artifact category: {category}")
    root = Path(cwd).resolve(strict=False) / category
    if name is None:
        return root
    # Deliberately require simple relative filenames; no arbitrary directories.
    if not name or "/" in name or "\\" in name or name in {".", ".."} or Path(name).is_absolute():
        raise ValueError(f"Unsafe artifact filename: {name!r}")
    return root / name


def task_tmp_dir(cfg, session: str) -> Path:
    """For pre-run tmux/stdout logs, which must survive run-leaf cleanup."""
    name = _component(session, "session")
    folder = task_root(cfg) / "_tmp" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def create_analysis_group(
    cfg, cwd: str | Path, relative_path: str, *, context_ids: list[str], description: str
) -> Path:
    """Initialize a scoped analytical subgroup; every leaf uses typed artifacts.

    This generic DLBase operation does not read the Knowledge graph at runtime.
    The offline harness/experiments/validate.py checks IDs against authoritative
    doc/knowledge/context_graph.json before a result is considered compliant.
    """
    run_dir = check_run_location(cfg, cwd)
    if not isinstance(relative_path, str):
        raise ValueError("Analysis group path must be a relative slash-separated string")
    components = relative_path.split("/")
    if not 1 <= len(components) <= 6 or any(not PART.fullmatch(c) for c in components):
        raise ValueError(f"Invalid analysis group path: {relative_path!r}")
    if not isinstance(context_ids, list) or not context_ids:
        raise ValueError("Analysis groups require at least one canonical Knowledge context ID")
    if any(
        not isinstance(x, str) or not re.fullmatch(r"[qsg]\.[A-Za-z0-9_.-]+", x)
        for x in context_ids
    ):
        raise ValueError("Invalid Knowledge ID; expected q.*, s.* or g.*")
    if len(set(context_ids)) != len(context_ids):
        raise ValueError("Duplicate Knowledge ID in analysis group")
    if (
        not isinstance(description, str)
        or not description.startswith("目标：")
        or len(description.strip()) < 12
    ):
        raise ValueError(
            "Analysis description must start with '目标：' and be at least 12 characters"
        )
    if len(description) > 600:
        raise ValueError("Analysis description exceeds 600 characters")
    group = run_dir / "analysis" / Path(*components)
    for folder in ARTIFACT_DIRS:
        (group / folder).mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": "analysis_group",
        "path": relative_path,
        "context_ids": context_ids,
        "description": description.strip(),
        "status": "initialized",
        "reference": "doc/knowledge/context_graph.json",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "artifacts": ARTIFACT_DIRS,
    }
    group_meta = group / "metadata/context.json"
    if group_meta.exists():
        prior = json.loads(group_meta.read_text(encoding="utf-8"))
        if (
            prior.get("context_ids") != context_ids
            or prior.get("description") != description.strip()
        ):
            raise FileExistsError(
                "Existing analysis context differs; do not silently overwrite its meaning"
            )
    else:
        tmp = group_meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(group_meta)
    manifest_path = run_dir / "metadata/run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = manifest.setdefault("analysis_groups", [])
        if relative_path not in groups:
            groups.append(relative_path)
            groups.sort()
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            tmp.replace(manifest_path)
    return group


def study_root(workspace: str | Path, study_id: str) -> Path:
    """Root for cross-run study manifests, frozen cohorts and publication staging.

    Individual Task results MUST continue to use run_leaf(cfg), including all
    metrics, intervention evidence, checkpoints and their typed directories.
    This root is for a study's *aggregate* metadata, not a new Task layout.
    """
    return Path(workspace).resolve() / "experiment" / "_studies" / _component(
        study_id, "study_id"
    )
