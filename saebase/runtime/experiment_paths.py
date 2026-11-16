"""Task-hierarchical experiment artifact contract used by every SAEBase task.

Path: experiment/<dataset>/<model>/<task>/<subtask...>/<sub_name>_<seed>/
Task-level temporary logs: experiment/<dataset>/<model>/<task>/<subtask...>/_tmp/<session>/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import fcntl
from contextlib import contextmanager
from typing import Any

PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
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
    components = run_components(cfg)
    return base / components["dataset"] / components["model"] / Path(*task_components(cfg))


def run_components(cfg) -> dict[str, str]:
    """Name artifacts from global inputs; tasks do not control the shared hierarchy."""
    from saebase.registry import selected_models
    from saebase.store import fingerprint

    names = sorted(item.name for item in selected_models(cfg))
    values = {
        "model": names[0] if len(names) == 1 else f"model_set-{fingerprint(names)[:12]}",
        "dataset": str(cfg.dataset.name),
    }
    return {name: _component(values[name], name) for name in ("dataset", "model")}


def hydra_run_component(name: str, *, _root_):
    if name not in {"dataset", "model"}:
        raise ValueError("Run component must be dataset or model")
    return run_components(_root_)[name]


def run_leaf(cfg) -> Path:
    sub = _component(cfg.experiment.sub_name, "experiment.sub_name")
    seed = _component(cfg.seed, "seed")
    return task_root(cfg) / f"{sub}_{seed}"


def check_run_location(cfg, cwd: str | Path) -> Path:
    if run_leaf(cfg).resolve(strict=False) != run_leaf(cfg):
        raise ValueError("Run location contains a symlink")
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
        and run_mode(cfg) == "overwrite"
        and not bool(cfg.get("runtime", {}).get("clear_experiment_leaf", True))
    ):
        raise ValueError("Overwriting an existing run requires runtime.clear_experiment_leaf=true")
    if prior and run_mode(cfg) == "new":
        shown = ", ".join(p.name for p in prior[:5])
        raise FileExistsError(
            f"Experiment leaf contains artifacts ({shown}); pick a new sub_name/seed, "
            "or explicitly set experiment.overwrite=true to replace them."
        )


def prepare_run(cfg, cwd: str | Path, *, note: str = "") -> dict:
    """Apply the reproducible artifact contract for every Task at SAEBase entry.

    Reused run leaves fail closed by default. Even for explicit overwrites,
    cleanup is limited to this one validated leaf, preserving Hydra .hydra/.
    """
    folder = check_run_location(cfg, cwd)
    mode = run_mode(cfg)
    identity = run_identity(cfg)
    guard_existing_run(cfg, folder)
    if mode == "resume":
        from saebase.runtime.experiment_results import _atomic

        dest = folder / "metadata" / "manifest.json"
        if not dest.is_file():
            raise FileNotFoundError("Resume requires an existing run manifest")
        previous = json.loads(dest.read_text())
        if previous.get("resume_identity") != identity:
            raise ValueError("Resume identity mismatch: configuration, inputs or source changed")
        attempts = list(previous.get("resume_attempts", []))
        from datetime import datetime, timezone

        attempts.append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "timeout_seconds": cfg.task.get("timeout_seconds"),
                "previous_status": previous.get("status"),
                "previous_error": previous.get("error"),
                "previous_completion": previous.get("completion"),
            }
        )
        previous["resume_attempts"] = attempts
        _atomic(dest, previous)
        return previous
    facets = cfg.get("runtime", {})
    folder.mkdir(parents=True, exist_ok=True)
    if bool(facets.get("clear_experiment_leaf", True)):
        for item in folder.iterdir():
            if item.name == ".hydra":
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    result = create_layout(cfg, folder, note=note)
    result["resume_identity"] = identity
    from saebase.runtime.experiment_results import _atomic

    _atomic(folder / "metadata" / "manifest.json", result)
    return result


def run_mode(cfg) -> str:
    """One outer run intent; task.resume controls only task-owned ledger reuse."""
    configured = cfg.get("runtime", {}).get("mode")
    legacy = cfg.experiment.get("overwrite")
    if legacy is not None and type(legacy) is not bool:
        raise ValueError("experiment.overwrite must be a boolean")
    mode = configured or ("overwrite" if legacy else "new")
    if mode not in {"new", "resume", "overwrite"}:
        raise ValueError("runtime.mode must be new, resume or overwrite")
    if legacy is not None and configured is not None and (mode == "overwrite") != legacy:
        raise ValueError("runtime.mode conflicts with experiment.overwrite")
    if mode == "resume":
        from saebase.registry import get_task_plugin

        task = get_task_plugin(str(cfg.task.name)).task_class()
        if not task.supports_run_resume(cfg):
            raise ValueError("This task does not support outer run resume")
        if not cfg.task.get("resume", True):
            raise ValueError("runtime.mode=resume conflicts with task.resume=false")
    return mode


def run_identity(cfg) -> str:
    """Bind recovery to scientific configuration, input bytes and scientific source."""
    from omegaconf import OmegaConf
    from tasks.sae_runtime import input_path

    value = OmegaConf.to_container(cfg, resolve=True)
    value.get("runtime", {}).pop("mode", None)
    value.get("experiment", {}).pop("overwrite", None)
    value["task"].pop("resume", None)
    value["task"].pop("timeout_seconds", None)
    inputs = {}
    if cfg.dataset.get("sample_file"):
        path = input_path(cfg.dataset.sample_file)
        inputs["dataset.sample_file"] = hashlib.sha256(path.read_bytes()).hexdigest()
        value["dataset"]["sample_file"] = str(path)
    for field in ("benchmark_config", "discovery_file", "heldout_file"):
        raw = cfg.task.get(field)
        if raw:
            path = input_path(raw)
            inputs[field] = hashlib.sha256(path.read_bytes()).hexdigest()
            value["task"][field] = str(path)
    source = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for family in ("saebase", "tasks", "models", "datasets"):
        for path in sorted((root / family).rglob("*.py")):
            source.update(path.relative_to(root).as_posix().encode())
            source.update(path.read_bytes())
    payload = {"configuration": value, "inputs": inputs, "source": source.hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def run_session(cfg, cwd: str | Path):
    """Hold an advisory writer lock outside the run leaf, including during cleanup."""
    folder = check_run_location(cfg, cwd)
    locks = task_root(cfg) / "_tmp" / "run_locks"
    locks.mkdir(parents=True, exist_ok=True)
    path = locks / (hashlib.sha256(str(folder).encode()).hexdigest()[:24] + ".lock")
    if path.is_symlink():
        raise ValueError("Run lock must not be a symlink")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Run already has an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def create_layout(cfg, cwd: str | Path, *, note: str = "") -> dict:
    folder = check_run_location(cfg, cwd)
    for sub in ARTIFACT_DIRS:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf
    from saebase.runtime.experiment_results import configuration_identity

    configuration = OmegaConf.to_container(cfg, resolve=True)
    metadata = {
        "schema": "saebase.experiment",
        "status": "running",
        "configuration": configuration,
        **run_components(cfg),
        "run_components": run_components(cfg),
        "task": str(cfg.task.name),
        "task_hierarchy": task_components(cfg),
        "sub_name": str(cfg.experiment.sub_name),
        "seed": cfg.seed,
        "run_leaf": folder.name,
        "configuration_sha256": configuration_identity(configuration),
        "note": note
        or "Run metadata generated by SAEBase; experiment outputs must follow this schema",
        "artifacts": ARTIFACT_DIRS,
        "evaluation_caveat": "Check the task-specific dataset split note; validation may not be independent of test.",
    }
    dest = folder / "metadata" / "manifest.json"
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
