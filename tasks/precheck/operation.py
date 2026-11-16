"""Budgeted SAE/Task smoke orchestration; Setup never owns a test.

Results remain in the existing Runtime-owned experiment leaf. Task workers are
isolated so native generation/training cannot silently exceed wall-time budget.
"""

from __future__ import annotations

from contextvars import ContextVar

import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from typing import Any

from importlib.resources import files

from omegaconf import DictConfig, OmegaConf

from saebase.results import EvaluationResult
from tasks.precheck.report import blocked, error_result, summarize
from tasks.precheck.sae_checks import select_sites
from tasks.sae_runtime import TASK_NAMES


def _budget(cfg: DictConfig) -> tuple[int, int, int]:
    for key in ("max_seconds", "max_samples", "max_tokens"):
        if key in cfg.task and type(cfg.task[key]) is not int:
            raise ValueError("Precheck requires bounded integer budgets")
    values = (
        int(cfg.task.get("max_seconds", 120)),
        int(cfg.task.get("max_samples", 2)),
        int(cfg.task.get("max_tokens", 32)),
    )
    if not (1 <= values[0] <= 3600 and 1 <= values[1] <= 64 and 1 <= values[2] <= 512):
        raise ValueError("Precheck requires bounded max_seconds/max_samples/max_tokens")
    return values


def _selected_tasks(cfg: DictConfig, mode: str) -> list[str]:
    if mode == "sae":
        return []
    given = list(cfg.task.get("check_tasks", []))
    tasks = [str(name) for name in given] if given else list(TASK_NAMES)
    if len(set(tasks)) != len(tasks) or any(name not in TASK_NAMES for name in tasks):
        raise ValueError(f"check_tasks must be unique members of {list(TASK_NAMES)}")
    return tasks


def _assets(
    cfg: DictConfig, sites: list[dict[str, Any]], tasks: list[str]
) -> list[dict[str, str | None]]:
    """Include Task-owned SAE requirements even in mode=task without SAE-only checks."""
    from saebase.api.sae_profiles import get_profile

    if not sites and (not tasks or all(name in {"train", "explain", "circuits"} for name in tasks)):
        # Native protocols own their model snapshots/generator/transcoder;
        # downloading the outer model label would prepare an unrelated resource.
        return []
    if "members" in cfg.model:
        entries = []
        for member in cfg.model.members:
            selected = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            selected.model = member
            entries.extend(_assets(selected, sites, tasks))
        return entries
    entries: list[dict[str, str | None]] = []
    model_key = str(cfg.model.name)
    from saebase.extensions.registry import external_model

    if external_model(model_key) is not None:
        return []
    model_repo = str(
        cfg.model.get("hub_repo_id")
        or cfg.model.get("hf_id")
        or cfg.model.get("repo_id")
        or cfg.model.get("pretrained_name")
        or ""
    )
    if "/" not in model_repo:
        from models.plugins import get_model_adapter

        module = get_model_adapter(model_key)
        model_repo = module.get_spec().model_name
    if not cfg.task.get("resource_only", False):
        entries.append(
            {
                "kind": "model",
                "model_key": model_key,
                "repo_id": model_repo,
                "revision": cfg.model.get("revision"),
            }
        )

    required_saes = [(str(site["sae_key"]), int(site["layer"])) for site in sites]
    configs = cfg.task.get("task_configs", {})
    for task_name in tasks:
        if task_name not in ("capture", "intervene", "generate"):
            continue
        own_config = configs.get(task_name, {})
        from tasks.sae_runtime import resolve_sae

        selected = resolve_sae(
            cfg.model, own_config.get("sae_key"), own_config.get("layer"), operation=task_name
        )
        required_saes.append((selected.sae_key, selected.layer))

    for sae_key, layer in dict.fromkeys(required_saes):
        profile = get_profile(sae_key)
        if model_key not in profile.allowed_model_keys():
            raise ValueError(f"Selected SAE {sae_key!r} does not support Model {model_key!r}")
        repo_id = str(profile.repo_id or "")
        from saebase.api.sae_loading import _phase_a_sae_lens_source

        release, source_id, _ = _phase_a_sae_lens_source(profile, layer)
        entries.append(
            {
                "kind": "sae",
                "repo_id": repo_id or release,
                "revision": None,
                "loader": profile.loader,
                "release": release,
                "sae_id": source_id,
                "folder": profile.sae_folder(layer) or source_id,
            }
        )

    unique = {}
    for entry in entries:
        unique[json.dumps(entry, sort_keys=True)] = entry
    return list(unique.values())


_WORKER_MODE = ContextVar("saebase_precheck_worker_mode", default="isolated")


def _run_worker(
    request: dict[str, Any], *, label: str, timeout: int, allow_network: bool
) -> dict[str, Any]:
    from saebase.runtime.experiment_paths import ARTIFACT_DIRS

    root = Path.cwd() / "tmp" / "precheck_jobs" / label
    for sub in ARTIFACT_DIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    inp = root / "metadata" / "request.json"
    out = root / "metadata" / "result.json"
    with inp.open("x", encoding="utf-8") as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if _WORKER_MODE.get() == "in_process":
        if not allow_network and request["kind"] not in {"dataset"}:
            raise ValueError("In-process network isolation is unavailable; use isolated mode")
        from tasks.precheck.worker import check
        import time

        from contextlib import redirect_stdout, redirect_stderr

        started = time.monotonic()
        previous = Path.cwd()
        try:
            os.chdir(root)
            with (
                (root / "logs/stdout.log").open("w") as stdout,
                (root / "logs/stderr.log").open("w") as stderr,
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    data = check(request)
        finally:
            os.chdir(previous)
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            data = blocked(
                stage=label,
                code="BLOCKED_RESOURCE",
                reason=f"Check exceeded {timeout}s",
                repair="Inspect batch wall-time budget",
            )
        data = {
            **data,
            "worker_mode": "in_process",
            "hard_kill_isolation": False,
            "elapsed_seconds": elapsed,
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return {**data, "stage": label, "receipt": str(out.relative_to(Path.cwd()))}
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    if not allow_network:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    command = [
        sys.executable,
        "-m",
        "tasks.precheck.worker",
        "--input",
        str(inp),
        "--output",
        str(out),
    ]
    try:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        (root / "logs" / "stdout.log").write_text(stdout, encoding="utf-8")
        (root / "logs" / "stderr.log").write_text(stderr, encoding="utf-8")
        return blocked(
            stage=label,
            code="BLOCKED_RESOURCE",
            reason=f"Real worker exceeded {timeout}s",
            attempted=["isolated worker started", "wall-time budget enforced"],
            repair="Provide enough compute or explicitly raise the bounded Precheck time budget.",
        )
    (root / "logs" / "stdout.log").write_text(stdout, encoding="utf-8")
    (root / "logs" / "stderr.log").write_text(stderr, encoding="utf-8")
    if not out.is_file():
        return {
            "status": "FAIL",
            "stage": label,
            "code": "WORKER_NO_RECEIPT",
            "reason": f"Worker exited {proc.returncode} without a result receipt; stderr: {stderr[-500:]}",
            "attempted": ["isolated worker launched"],
            "repair": "Inspect the worker logs and dependency environment.",
            "retryable": False,
        }
    data = json.loads(out.read_text(encoding="utf-8"))
    if data.get("status") not in ("PASS", "FAIL", "BLOCKED"):
        return error_result(
            ValueError("Invalid worker result state"),
            stage=label,
            attempted=["read worker receipt"],
        )
    return {
        **data,
        "stage": label,
        "receipt": str(out.relative_to(Path.cwd())),
        "exit_code": proc.returncode,
    }


def _task_config(cfg: DictConfig, name: str, max_seconds: int, max_tokens: int) -> dict[str, Any]:
    path = files("conf") / "task" / f"{name}.yaml"
    original = OmegaConf.load(path)
    task = dict(cfg.task.get("task_configs", {}).get(name, {}))
    # Hydra's structured Precheck node does not contain the target Task's fields.
    # Replace that node with the Task-owned schema rather than merging into it.
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    base.task = original.task
    target = OmegaConf.merge(
        base,
        {
            "task": {
                "name": name,
                "execute": True,
                "device": str(cfg.task.get("device", "cpu")),
                "prompt": str(cfg.task.get("prompt", "")),
                "timeout_seconds": max_seconds,
                "extensions": list(cfg.task.get("extensions", [])),
            }
        },
        {"task": task},
    )
    if name == "generate":
        specified = int(target.task.get("max_new_tokens", max_tokens))
        if specified > max_tokens:
            raise ValueError("Task generation exceeds max_tokens budget")
        target.task.max_new_tokens = specified
    if name in ("benchmark", "circuits", "train", "explain") and not task:
        raise ValueError(
            f"Task {name} needs an explicit, bounded task.task_configs.{name} plan "
            "before a real smoke; Precheck must not run an unbounded default protocol"
        )
    if name in ("capture", "intervene", "generate"):
        from tasks.sae_runtime import resolve_sae

        selected = resolve_sae(
            target.model, target.task.get("sae_key"), target.task.get("layer"), operation=name
        )
        target.task.sae_key = selected.sae_key
        target.task.layer = selected.layer
    return OmegaConf.to_container(target, resolve=True)


def run_precheck(cfg: DictConfig) -> EvaluationResult:
    """Default isolated checks; explicit trusted batches may reuse a live pair.

    In-process mode has no per-worker hard-kill isolation. Its caller must enforce
    a whole-process wall-time bound; receipts explicitly preserve this distinction.
    """
    mode = str(cfg.task.get("worker_mode", "isolated"))
    if mode not in {"isolated", "in_process"}:
        raise ValueError("Precheck worker_mode must be isolated or in_process")
    token = _WORKER_MODE.set(mode)
    try:
        return _run_precheck(cfg)
    finally:
        _WORKER_MODE.reset(token)


def _run_precheck(cfg: DictConfig) -> EvaluationResult:
    from models.loading_policy import resolve_config_devices

    resolve_config_devices(cfg)
    mode = str(cfg.task.get("mode", "sae"))
    rows: list[dict[str, Any]] = []
    preparation: list[dict[str, Any]] = []
    from saebase.runtime.experiment_paths import run_components

    model_key = run_components(cfg)["model"]
    dataset_key = str(cfg.dataset.name)

    if mode not in ("sae", "task", "all"):
        raise ValueError("Precheck mode must be sae, task or all")
    max_seconds, max_samples, max_tokens = _budget(cfg)
    tasks = _selected_tasks(cfg, mode)
    if not bool(cfg.task.get("execute", False)):
        rows.append(
            {
                "status": "NOT_RUN",
                "stage": "precheck",
                "reason": "Set task.execute=true to perform any real checks.",
            }
        )
    else:
        sites: list[dict[str, Any]] = []
        if mode in ("sae", "all"):
            try:
                sites = select_sites(cfg.model, cfg.task.get("sae_list", "default"))
                if len(sites) > max_samples:
                    raise ValueError("Selected SAE sites exceed the max_samples check budget")
            except Exception as exc:
                rows.append(
                    error_result(
                        exc, stage="sae_selection", attempted=["read Model-owned SAE choices"]
                    )
                )
        setup_ok = True
        from saebase.extensions.registry import external_model

        is_external = external_model(model_key) is not None
        if is_external and not cfg.task.get("extensions") and _WORKER_MODE.get() != "in_process":
            setup_ok = False
            rows.append(
                blocked(
                    stage="extension_worker",
                    code="BLOCKED_CONFIGURATION",
                    reason="External Precheck workers require explicit installed task.extensions names",
                    repair="Provide installed extension entry-point names for the isolated worker.",
                )
            )
        if setup_ok and bool(cfg.task.get("auto_setup", True)):
            try:
                parameters = dict(
                    required_modules=[]
                    if is_external
                    else (
                        ("huggingface_hub", "sae_lens", "transformer_lens", "transformers")
                        if (sites or any(t in ("capture", "intervene", "generate") for t in tasks))
                        else ("huggingface_hub",)
                        if tasks
                        else ()
                    ),
                    assets=_assets(cfg, sites, tasks),
                    allow_network=bool(cfg.task.get("allow_network", True)),
                    allow_install=bool(cfg.task.get("allow_install", True)),
                    allow_download=bool(cfg.task.get("allow_download", True)),
                    max_download_bytes=int(cfg.task.get("max_download_bytes", 16 * 1024**3)),
                    install_timeout_seconds=max(
                        1, int(cfg.task.get("prepare_timeout_seconds", 1800)) - 30
                    ),
                )
                if (
                    ("capture" in tasks or cfg.task.get("target_operation") == "capture")
                    and not is_external
                    and cfg.task.get("task_configs", {})
                    .get("capture", {})
                    .get("neuronpedia", {})
                    .get("enabled", True)
                ):
                    parameters["required_modules"] = [
                        *parameters["required_modules"],
                        "neuronpedia",
                    ]
                prepared = _run_worker(
                    {"kind": "prepare_assets", "parameters": parameters},
                    label="setup",
                    timeout=int(cfg.task.get("prepare_timeout_seconds", 1800)),
                    allow_network=parameters["allow_network"],
                )
                preparation.append(prepared)
                if prepared["status"] != "PASS":
                    rows.append(prepared)
                    setup_ok = False
            except Exception as exc:
                rows.append(
                    error_result(exc, stage="setup", attempted=["Setup preparation requested"])
                )
                setup_ok = False
        device = str(cfg.task.get("device", "cpu"))
        whiten_basis = (
            cfg.task.get("target_operation") == "capture"
            and cfg.task.get("task_configs", {}).get("capture", {}).get("mode", "non_whiten")
            == "whiten"
        )
        needs_dataset = (
            cfg.dataset.get("samples") is None
            and not cfg.task.get("resource_only", False)
            and (not cfg.task.get("prompt") or whiten_basis)
            and (sites or any(name in {"capture", "intervene", "generate"} for name in tasks))
        )
        if setup_ok and needs_dataset and bool(cfg.task.get("auto_setup", True)):
            dataset_cfg = OmegaConf.merge(cfg, {"model": {"device": device}})
            prepared = _run_worker(
                {
                    "kind": "prepare_dataset",
                    "config": OmegaConf.to_container(dataset_cfg, resolve=True),
                    "n_generate": int(
                        cfg.dataset.get("sample_count", cfg.task.get("dataset_n_generate", 32))
                    ),
                    "min_kept": int(cfg.task.get("dataset_min_kept", 1)),
                },
                label="dataset_prepare",
                timeout=int(cfg.task.get("dataset_prepare_timeout_seconds", 1800)),
                allow_network=bool(cfg.task.get("allow_network", True)),
            )
            preparation.append(prepared)
            if prepared["status"] != "PASS":
                rows.append(prepared)
                setup_ok = False
        if setup_ok and needs_dataset:
            checked = _run_worker(
                {
                    "kind": "dataset",
                    "dataset_key": dataset_key,
                    "model_key": model_key,
                    "data_root": str(cfg.paths.data_dir),
                    "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
                    "max_records": int(cfg.task.get("check_sample_count", 32)),
                    "sampling_seed": int(cfg.task.get("sampling_seed", 42)),
                },
                label="dataset_check",
                timeout=max_seconds,
                allow_network=False,
            )
            rows.append(checked)
            setup_ok = checked["status"] == "PASS"
        if setup_ok:
            prompt = str(cfg.task.get("prompt", ""))
            probes = [{"prompt": prompt, "sample_id": "explicit_prompt"}]
            if not prompt and needs_dataset:
                from saebase.datasets.prompts import read_prompts
                from saebase.datasets.sampling import sample_records

                available = read_prompts(cfg.dataset, model_key=model_key)
                probes = sample_records(
                    len(available),
                    available.__getitem__,
                    int(cfg.task.get("check_sample_count", 32)),
                    int(cfg.task.get("sampling_seed", 42)),
                )
                prompt = probes[0]["prompt"]
            for index, site in enumerate(sites):
                request = {
                    "kind": "sae_checkpoint" if cfg.task.get("resource_only", False) else "sae",
                    "operation": str(cfg.task.get("target_operation", "precheck")),
                    "model_key": model_key,
                    "sae_key": site["sae_key"],
                    "layer": site["layer"],
                    "prompt": prompt,
                    "prompts": [row["prompt"] for row in probes],
                    "sample_ids": [row.get("sample_id", row.get("id")) for row in probes],
                    "device": device,
                    "extensions": list(cfg.task.get("extensions", [])),
                }
                if cfg.task.get("target_operation") == "capture":
                    from tasks.capture.operation import neuronpedia_settings

                    annotation_cfg = OmegaConf.create({"task": cfg.task.task_configs.capture})
                    request["annotation_options"] = neuronpedia_settings(annotation_cfg)
                    request["capture_mode"] = str(annotation_cfg.task.get("mode", "non_whiten"))
                    request["whiten_chunk_size"] = int(
                        annotation_cfg.task.get("whiten_chunk_size", 256)
                    )
                rows.append(
                    _run_worker(
                        request,
                        label=f"sae_{index:02d}",
                        timeout=max_seconds,
                        allow_network=bool(cfg.task.get("allow_network", True)),
                    )
                )
            for index, name in enumerate(tasks):
                try:
                    task_cfg = _task_config(cfg, name, max_seconds, max_tokens)
                    request = {
                        "kind": "task_preparation"
                        if name in {"benchmark", "circuits", "explain", "train"}
                        or cfg.dataset.get("samples") is not None
                        else "task",
                        "operation": name,
                        "config": task_cfg,
                        "extensions": list(cfg.task.get("extensions", [])),
                    }
                    rows.append(
                        _run_worker(
                            request,
                            label=f"task_{index:02d}_{name}",
                            timeout=max_seconds,
                            allow_network=bool(cfg.task.get("allow_network", True)),
                        )
                    )
                except Exception as exc:
                    rows.append(
                        error_result(
                            exc,
                            stage=f"task:{name}",
                            attempted=["validate bounded Task configuration"],
                        )
                    )
    summary = summarize(rows)
    payload = {
        "schema": "saebase.precheck.v1",
        "model": model_key,
        "dataset": dataset_key,
        "mode": mode,
        "budget": {
            "max_seconds_per_check": max_seconds,
            "max_samples": max_samples,
            "max_tokens": max_tokens,
        },
        "rows": rows,
        "preparation": preparation,
        "summary": summary,
        "setup_is_not_a_test": True,
    }
    destination = Path.cwd() / "metadata" / "precheck_result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        from saebase.runtime.audit import audit_artifacts, file_sha256

        size = destination.stat().st_size
        manifest = {
            "schema": "saebase.artifact_manifest.v1",
            "files": [
                {
                    "path": "metadata/precheck_result.json",
                    "bytes": size,
                    "sha256": file_sha256(destination),
                    "required": True,
                }
            ],
        }
        audit = audit_artifacts(Path.cwd(), manifest)
        (Path.cwd() / "metadata" / "precheck_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if audit["status"] != "PASS":
            summary["status"] = "FAIL"
    except Exception as exc:
        summary["status"] = "FAIL"
        rows.append(
            error_result(exc, stage="artifact_audit", attempted=["audit real report bytes"])
        )
    return EvaluationResult(
        ok=summary["status"] == "PASS",
        task="precheck",
        status=(
            "configured_only"
            if summary["status"] == "NOT_RUN"
            else "blocked"
            if summary["status"] == "BLOCKED"
            else "failed"
            if summary["status"] == "FAIL"
            else "executed"
        ),
        evaluation=mode,
        model_key=model_key,
        prompt_source=dataset_key,
        summary=summary,
        per_prompt=[],
        meta=payload,
        error=None
        if summary["status"] == "PASS"
        else "Some requested checks were not verified; see metadata/precheck_result.json",
    )
