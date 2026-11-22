"""Purpose and main functions: Record, discover, read and compare local experiment results.

Scope: Runtime-owned metadata and task-owned metric ledgers inside explicit run roots.
The functions do not execute tasks, alter measurements or infer cache validity from names.
"""

from __future__ import annotations

from conf import configuration_root

from saebase.runtime.configuration.task_options import task_value
import hashlib
import copy
import warnings
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from omegaconf import OmegaConf


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".run-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _configuration(cfg: Any) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)


write_json = _atomic


def historical_run_path(folder: str | Path) -> Path:
    """Locate a moved experiment path when its published location has no entry."""
    root = Path(folder).resolve()
    if root.exists():
        return root
    for parent in (root, *root.parents):
        if parent.name != "experiment":
            continue
        archive = parent.parent / "assets/history_experiment/local_runs/files/experiment"
        saved = archive / root.relative_to(parent)
        if saved.exists():
            return saved
    return root


def _source() -> dict[str, Any]:
    root = configuration_root().parent
    if not (root / ".git").exists():
        return {"commit": None, "working_tree": None}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    values = {}
    for name, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("working_tree", ["status", "--porcelain"]),
    ):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], env=env, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            values[name] = None
        else:
            values[name] = result.stdout.strip() if result.returncode == 0 else None
    return values


def configuration_identity(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _result_paths(payload: dict, root: Path, *, reading=False) -> dict:
    """Store internal references relative to their run and resolve them for consumers."""

    value = copy.deepcopy(payload)
    root = root.resolve()

    def reference(raw):
        path = Path(raw)
        if reading and path.is_absolute():
            path = historical_run_path(path)
        if reading:
            # Earlier runs used absolute result paths or placed automatic precheck under tmp/.
            local = None
            previous_precheck = Path("tmp/automatic_precheck/metadata/precheck_result.json")
            if (
                path.parts[-len(previous_precheck.parts):] == previous_precheck.parts
                and (not path.is_absolute() or not path.exists())
            ):
                local = root / "metadata/automatic_precheck/metadata/precheck_result.json"
            elif (
                path.is_absolute()
                and not path.is_relative_to(root)
                and path.parent.name == "metadata"
                and path.name == f"{value.get('task')}_result.json"
            ):
                local = root / "metadata" / path.name
            if local is not None and local.resolve().is_relative_to(root) and local.is_file():
                return str(local)
        if path.is_absolute():
            if not path.is_relative_to(root):
                return str(path)
            path = path.relative_to(root)
        return str(root / path) if reading else path.as_posix()

    value["artifact_paths"] = {
        key: reference(raw) for key, raw in value.get("artifact_paths", {}).items()
    }
    summary = value.get("summary", {})
    for key in ("result_path", "samples_path", "explanations_path"):
        if summary.get(key):
            summary[key] = reference(summary[key])
    for item in summary.get("required_artifacts", []):
        item["path"] = reference(item["path"])
    return value


def record_run(folder: str | Path, cfg: Any, *, result=None, error: dict | None = None) -> dict:
    """Write current run metadata while preserving the original operation payload."""
    root = Path(folder)
    config = _configuration(cfg)
    payload = _result_paths(result.to_dict(), root) if result is not None else None
    summary = result.summary if result is not None else {}
    from saebase.runtime.experiment_paths import run_components
    from saebase.runtime.environment import snapshot_environment

    components = run_components(cfg)
    # A task may supply a complete, verified identity. Configuration alone never enables reuse.
    identity = summary.get("input_identity")
    previous_path = root / "metadata" / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    previous.pop("config_sha256", None)
    manifest = {
        **previous,
        "schema": "saebase.experiment",
        "task": str(cfg.task.name),
        "task_mode": task_value(config["task"], "mode"),
        "dataset": summary.get("resolved_protocol", {}).get("dataset") or components["dataset"],
        "model": result.model_key if result is not None else components["model"],
        "seed": cfg.seed,
        "configuration": config,
        "original_configuration": previous.get(
            "original_configuration", previous.get("configuration", config)
        ),
        "configuration_sha256": configuration_identity(config),
        "source": _source(),
        "environment": snapshot_environment(),
        "status": "failed" if error is not None else result.status,
        "format_version": 2,
        "completion": result.completion if result is not None else {},
        "exit_code": result.exit_code if result is not None else 1,
        "run_components": components,
        "result": payload,
        "error": error
        if error is not None
        else (
            {"type": "TaskFailure", "message": result.error}
            if result is not None and result.error
            else None
        ),
        "input_identity": identity,
        "artifacts": {
            name: name
            for name in ("metrics", "checkpoints", "tables", "figures", "logs", "reports")
            if (root / name).is_dir()
        },
    }
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    if previous.get("effective_config"):
        config["component_paths"] = previous["configuration"]["component_paths"]
        status = manifest["status"]
        manifest["task_status"] = status
        completed = status in {"configured_only", "executed"} and not manifest["exit_code"]
        manifest["status"] = "succeeded" if completed else "failed"
        summary_path = root / "reports/task_result.json"
        _atomic(
            summary_path, {"task": manifest["task"], "status": status, "exit_code": manifest["exit_code"]}
        )
        manifest["outputs"] = {"task_result": summary_path.relative_to(root).as_posix()}
        manifest["configuration_sha256"] = configuration_identity(config)
        if config != previous["configuration"]:
            manifest["effective_config"] = "metadata/final_configuration.json"
            _atomic(root / manifest["effective_config"], config)
    if not (metadata / "config.yaml").exists():
        (metadata / "config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(config), resolve=True)
        )
    _atomic(metadata / "manifest.json", manifest)
    (metadata / "latest_configuration.json").unlink(missing_ok=True)
    return manifest


def archive_execution(folder: Path, artifacts: list[Path]) -> Path:
    """Commit a completed execution to a file before replacing its live artifacts."""
    import tarfile

    manifest_path = folder / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    execution_id = manifest.get("execution_id") or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{32}(?:[0-9a-f]{32})?", execution_id):
        raise ValueError("Invalid execution record identifier")
    target = folder / "executions" / f"{execution_id}.tar"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        return target
    temporary = target.with_suffix(".writing")
    try:
        with tarfile.open(temporary, "w") as archive:
            for artifact in artifacts:
                archive.add(artifact, arcname=artifact.name)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _execution_folders(root):
    return [
        path for path in sorted((root / "executions").glob("*"))
        if (path / "metadata/manifest.json").is_file()
    ]


def list_executions(folder: str | Path) -> list[dict]:
    """List saved configurations and artifact locations at one named experiment."""
    import tarfile

    root = historical_run_path(folder)
    results = []
    for path in sorted((root / "executions").glob("*.tar")):
        with tarfile.open(path, "r") as archive:
            record = json.load(archive.extractfile("metadata/manifest.json"))
        results.append({
            **record, "execution_id": path.stem, "storage": str(path), "current": False,
        })
    manifest = root / "metadata/manifest.json"
    if manifest.is_file():
        results.append({**json.loads(manifest.read_text()), "storage": str(root), "current": True})
    executions = _execution_folders(root)
    for execution in executions:
        record = json.loads((execution / "metadata/manifest.json").read_text())
        record["status"] = record.get("task_status", record["status"])
        results.append({**record, "storage": str(execution), "current": execution == executions[-1]})
    return results


def read_run(folder: str | Path) -> dict:
    """Read a run; its referenced task ledgers remain responsible for their own integrity."""

    root = historical_run_path(folder)
    if not (root / "metadata/manifest.json").is_file() and (executions := _execution_folders(root)):
        root = executions[-1]
    manifest = json.loads((root / "metadata" / "manifest.json").read_text())
    if manifest.get("schema") != "saebase.experiment":
        raise ValueError("Unsupported experiment manifest")
    if "task_status" in manifest:
        manifest["status"] = manifest["task_status"]
    parameters = manifest.get("configuration", {}).get("task", {})
    try:
        from saebase.registry import get_task_plugin

        task = get_task_plugin(manifest["task"]).task_class()
    except (KeyError, ValueError):
        # Unknown historical Tasks remain browsable without inventing a mode.
        pass
    else:
        parameters = task.continuation_configuration(manifest["configuration"])["task"]
        manifest = task.read_result(root, manifest)
    if manifest.get("result") is not None:
        from saebase.results import EvaluationResult

        projected = _result_paths(manifest["result"], root, reading=True)
        projected.get("meta", {}).pop("execution_status", None)
        projected.get("meta", {}).pop("executed", None)
        manifest["result"] = EvaluationResult(**projected).to_dict()
    else:
        from saebase.results import result_diagnostics

        error = manifest.get("error") or {}
        manifest["diagnostics"] = result_diagnostics(
            manifest["task"], manifest.get("status"), {}, {}, {}, error.get("message")
        )
    return {**manifest, "task_mode": task_value(parameters, "mode"), "path": str(root.resolve())}


def inspect_run(folder: str | Path, *, expected_cfg=None) -> dict:
    """Determine continuation from persisted results and Task policy, using metadata only.

    This inspects the original run's completion, not the reusability of its
    scientific artifacts in a new experiment. No model or ledger is loaded.
    """
    from saebase.registry import get_task_plugin
    from saebase.results import EvaluationResult


    root = Path(folder).resolve()
    try:
        manifest = read_run(root)
        task = get_task_plugin(manifest["task"]).task_class()
        if expected_cfg is not None:

            def comparable(cfg):
                value = _configuration(cfg)
                runtime = value.pop("runtime", {})
                # Mode/cleanup are operational. Arithmetic changes the
                # scientific result and must not be ignored when resuming.
                numerics = None
                if runtime.get("numerics", "native") != "native":
                    numerics = {
                        "numerics": runtime["numerics"],
                        "numerics_protocol": runtime.get("numerics_protocol"),
                    }
                value.pop("experiment", None)
                value.pop("exp_run_leaf", None)
                value.pop("component_paths", None)
                task_config = value.get("task", {})
                if task_config.get("path") == task_config.get("name"):
                    task_config.pop("path", None)
                return task.continuation_configuration(value), numerics

            if comparable(manifest["configuration"]) != comparable(expected_cfg):
                return {
                    "action": "retry",
                    "state": "incomplete",
                    "reason": "configuration_mismatch",
                    "path": str(root),
                }
        raw = manifest.get("result")
        if raw is None:
            return {
                "action": "retry",
                "state": "failed" if manifest.get("status") == "failed" else "incomplete",
                "reason": "no_task_result",
                "path": str(root),
                "diagnostics": manifest.get("diagnostics", []),
            }
        result = EvaluationResult(**raw)
        if manifest["status"] != result.status or manifest["exit_code"] != result.exit_code:
            raise ValueError("Manifest and Task result status disagree")
        if (
            manifest["task"] != result.task
            or manifest["model"] != result.model_key
            or manifest["dataset"] != result.prompt_source
        ):
            raise ValueError("Manifest and Task result identity disagree")
        for artifact in task.required_result_artifacts(result):
            path = Path(artifact["path"])
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Required result artifact is missing or outside its run")
            if "bytes" in artifact:
                if type(artifact["bytes"]) is not int or artifact["bytes"] < 0:
                    raise ValueError("Invalid committed artifact byte count")
                if path.stat().st_size != artifact["bytes"]:
                    raise ValueError(
                        "Required result artifact size differs from its committed size"
                    )
        action = task.continuation_action(result)
        if action not in {"skip", "retry", "annotations_pending"}:
            raise ValueError("Unknown Task continuation action")
        state = (
            "complete"
            if action == "skip"
            else "annotations_pending"
            if action == "annotations_pending"
            else "failed"
            if result.status == "failed"
            else "blocked"
            if result.status in {"blocked", "unsupported"}
            else "not_run"
            if result.status == "configured_only"
            else "incomplete"
        )
        return {
            "action": action,
            "state": state,
            "reason": "task_completion_policy",
            "path": str(root),
            "status": result.status,
            "completion": result.completion,
            "exit_code": result.exit_code,
            "stages": result.summary.get("stages", {}),
            "diagnostics": result.diagnostics,
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "action": "retry",
            "state": "incomplete" if root.exists() else "not_run",
            "reason": "incomplete_run_record",
            "detail": str(exc),
            "path": str(root),
        }


def list_runs(root: str | Path, *, errors: list[dict] | None = None, **filters: Any) -> list[dict]:
    """List manifests under one explicit local root, with exact metadata filters."""
    allowed = {"task", "task_mode", "dataset", "model", "seed", "status"}
    if set(filters) - allowed:
        raise ValueError(f"Unknown result filters: {sorted(set(filters) - allowed)}")
    result = []
    root = historical_run_path(root)
    saved = root.parent / "assets/history_experiment/local_runs/files/experiment"
    roots = [root, saved] if root.name == "experiment" and saved.is_dir() else [root]
    restored = set()
    for path in (path for directory in roots for path in sorted(directory.rglob("metadata/manifest.json"))):
        if path in restored:
            continue
        try:
            item = read_run(path.parent.parent)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue = {
                "path": str(path),
                "reason": str(exc),
                "impact": "This run record is temporarily unread and the rest of the readable results continue to return.",
                "next_steps": ["Check the document and the mission information cited, and recheck it after repair."],
            }
            if errors is not None:
                errors.append(issue)
            else:
                warnings.warn(f"Unreadable run {path}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        if item.get("archived_manifest"):
            restored.add((root.parent / item["archived_manifest"]).resolve())
        if all(item.get(key) == value for key, value in filters.items()):
            result.append(item)
    return result


def find_reusable_runs(root: str | Path, input_identity: dict[str, Any]) -> list[dict]:
    """Find completed runs matching a caller-supplied complete measured input identity."""
    if not input_identity or not isinstance(input_identity, dict):
        raise ValueError(
            "Provide the exact verified input identity; names or config hashes are insufficient"
        )
    for key in ("model_sha256", "data_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(input_identity.get(key, ""))):
            raise ValueError(f"Verified input identity requires {key}")
    if "sae_sha256" in input_identity and not re.fullmatch(
        r"[0-9a-f]{64}", str(input_identity["sae_sha256"])
    ):
        raise ValueError("Invalid SAE content identity")
    return [
        {**item, "reuse_status": "candidate_only", "reuse_authorized": False}
        for item in list_runs(root, status="executed")
        if item.get("input_identity") == input_identity and item.get("exit_code", 0) == 0
    ]


def compare_runs(left: str | Path, right: str | Path, *, pairs=None) -> dict:
    """Delegate comparison semantics to the owning task; unsupported is explicit."""
    from saebase.registry import get_task_plugin

    a, b = read_run(left), read_run(right)
    if a["task"] != b["task"]:
        report = {"supported": False, "reason": "Runs belong to different tasks"}
    else:
        report = get_task_plugin(a["task"]).task_class().compare_results(left, right, pairs=pairs)
    return {
        "left": a,
        "right": b,
        "configuration_equal": a["configuration"] == b["configuration"],
        **report,
    }
