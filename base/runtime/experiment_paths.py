"""Task-hierarchical experiment artifact contract used by every SAEBase task.

Path: experiment/<dataset>/<model>/<task>/<subtask...>/<sub_name>/<run_id>/
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import sys
from uuid import uuid4
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conf import configuration_location, configuration_root, read_configuration
from saebase.runtime.configuration.task_options import task_owner, task_value

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
    for index, part in enumerate(parts[1:], start=1):
        experiment_name(part, Path(*parts[:index]), field="task.path")
    return parts


def project_root(cfg) -> Path:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd
    from omegaconf import OmegaConf

    settings = OmegaConf.to_container(cfg.paths, resolve=False) if OmegaConf.is_config(cfg.paths) else cfg.paths
    if settings["root"] == "${hydra:runtime.cwd}":
        return Path(get_original_cwd() if HydraConfig.initialized() else Path.cwd()).resolve()
    return Path(str(cfg.paths.root)).resolve()


def task_root(cfg) -> Path:
    base = project_root(cfg) / "experiment"
    components = run_components(cfg)
    hierarchy = Path(*task_components(cfg))
    if "models" in components:
        return base / "batches" / hierarchy
    return base / components["dataset"] / components["model"] / hierarchy


def run_components(cfg) -> dict[str, str | list[str]]:
    """Record the actual inputs, including the members of a joint model run."""
    from saebase.registry import selected_models

    names = sorted(_component(item.name, "model") for item in selected_models(cfg))
    dataset = _component(cfg.dataset.name, "dataset")
    if len(names) == 1:
        return {"dataset": dataset, "model": names[0]}
    return {"dataset": dataset, "model": "", "models": names}


def experiment_name(value: Any, parent: Path | None = None, *, field: str = "experiment.sub_name") -> str:
    """Validate a name that describes an experiment or an execution's distinct role."""
    name = str(value)
    words = r"[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){0,2}"
    placeholder = r"(?:run|seed|trial|repeat|job|study|experiment|basic|default|[a-z])(?:_?\d+)?"
    if name in ARTIFACT_DIRS or name in {"executions", "multirun"}:
        raise ValueError(
            f"{field} uses the reserved artifact directory {name!r}. "
            "Choose a name that describes the purpose, such as answer_stability."
        )
    if (
        len(name) > 96
        or not re.fullmatch(words, name)
        or (re.fullmatch(placeholder, name) is not None and name != "i")
        or re.search(r"(?:19|20)\d{6}|(?:^|_)seed\d*(?:_|$)", name)
    ):
        raise ValueError(
            f"{field} must contain one to three lowercase words describing a purpose, "
            "such as answer_stability or top10_candidates"
        )
    if parent is not None:
        for level in parent.parts:
            if name == level or name.startswith(f"{level}_"):
                raise ValueError(
                    f"{field} repeats {level!r}; use the added purpose, "
                    "such as answer_stability"
                )
    return name


def configured_experiment_name(cfg) -> str | None:
    hierarchy = Path(*task_components(cfg))
    name = cfg.experiment.get("sub_name")
    if name is not None:
        return experiment_name(name, hierarchy)
    mode = task_value(cfg.task, "mode") or task_value(cfg.task, "evaluation")
    if mode and str(mode).lower() not in hierarchy.parts:
        return experiment_name(str(mode).lower(), hierarchy)
    return None


def experiment_directory(task_directory: Path, name: str | None, run_id: str | None = None) -> Path:
    folder = task_directory / experiment_name(name, Path(task_directory.name)) if name else task_directory
    if run_id is not None:
        relative = folder.relative_to(task_directory.parent)
        folder = folder / experiment_name(run_id, relative, field="experiment.run_id")
    return folder


def member_directory(
    root: Path, dataset: str, model: str, task: str, name: str, role: str | None = None
) -> Path:
    """Derive a member's location from its actual inputs and research purpose."""
    directory = (
        root / "experiment" / _component(dataset, "dataset") / _component(model, "model")
        / _component(task, "task")
    )
    return experiment_directory(directory, name, role)


def capture_method_role(name: str, mode: str) -> str | None:
    method = mode.lower()
    return None if name == method else method


def matrix_report(root: Path, task: str, name: str, role: str | None = None) -> Path:
    """Locate a batch's coordination report, outside any individual run."""
    directory = root / "experiment" / "batches" / _component(task, "task")
    return experiment_directory(directory, name, role) / "reports" / "matrix"


def hydra_run_path(level: str, *, _root_) -> str:
    if level == "task":
        return str(task_root(_root_))
    if level == "leaf":
        return str(run_leaf(_root_))
    raise ValueError("Run path level must be task or leaf")


def run_leaf(cfg) -> Path:
    name = configured_experiment_name(cfg)
    role = cfg.experiment.get("run_id")
    if role is not None:
        hierarchy = Path(*task_components(cfg))
        experiment_name(role, hierarchy / name if name else hierarchy, field="experiment.run_id")
    return experiment_directory(task_root(cfg), name, role)


def check_run_location(cfg, cwd: str | Path) -> Path:
    if run_leaf(cfg).resolve(strict=False) != run_leaf(cfg):
        raise ValueError("Run location contains a symlink")
    expected = run_leaf(cfg).resolve(strict=False)
    actual = Path(cwd).resolve(strict=False)
    if expected == actual:
        return actual
    raise ValueError(f"Hydra output location mismatch: expected {expected}, got {actual}")


def _run_artifacts(folder: Path) -> list[Path]:
    return [folder / name for name in ARTIFACT_DIRS if (folder / name).exists()]


def guard_existing_run(cfg, cwd: str | Path) -> None:
    """Protect incomplete work and unrelated directories at a named experiment location."""
    folder = check_run_location(cfg, cwd)
    prior = _run_artifacts(folder)
    mode = run_mode(cfg)
    if prior and mode == "overwrite" and not bool(cfg.runtime.get("clear_experiment_leaf", True)):
        raise ValueError("Overwriting an existing run requires runtime.clear_experiment_leaf=true")
    if prior and mode == "new":
        manifest = folder / "metadata" / "manifest.json"
        if not manifest.is_file():
            raise FileExistsError("Experiment has unregistered artifacts. Choose a distinct experiment purpose.")
        status = json.loads(manifest.read_text()).get("status")
        if status == "running":
            raise FileExistsError("Experiment has an unfinished execution. Resume it or choose a distinct purpose.")


def prepare_run(cfg, cwd: str | Path, *, note: str = "") -> dict:
    """Preserve terminal executions before starting new work at the same named location."""
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
    folder.mkdir(parents=True, exist_ok=True)
    prior = _run_artifacts(folder)
    if prior:
        if mode == "new":
            from saebase.runtime.experiment_results import archive_execution

            archive_execution(folder, prior)
        for item in prior:
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

    from saebase.runtime.sae_operations import input_path

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
        raw = task_value(cfg.task, field)
        if raw:
            path = input_path(raw)
            physical, section = configuration_location(path) if field == "benchmark_config" else (path, "")
            content = (
                json.dumps(read_configuration(path, resolve=True), sort_keys=True).encode()
                if section
                else physical.read_bytes()
            )
            inputs[field] = hashlib.sha256(content).hexdigest()
            task_owner(value["task"])[field] = str(path)
    source = hashlib.sha256()
    root = configuration_root().parent
    source_directories = ("base", "datasets", "third_party") if (root / "base").is_dir() else ("saebase",)
    for family in (*source_directories, "tasks", "models"):
        paths = {path.relative_to(root): path for path in (root / family).rglob("*.py")}
        for logical_path, path in sorted(paths.items()):
            source.update(logical_path.as_posix().encode())
            source.update(path.read_bytes())
    payload = {"configuration": value, "inputs": inputs, "source": source.hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def run_session(cfg, cwd: str | Path):
    """Hold an advisory writer lock outside the run leaf, including during cleanup."""
    folder = check_run_location(cfg, cwd)
    locks = project_root(cfg) / "experiment" / ".locks"
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
        "sub_name": configured_experiment_name(cfg),
        "run_id": cfg.experiment.get("run_id"),
        "execution_id": uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
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
    folder = experiment_directory(task_root(cfg), name) / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
