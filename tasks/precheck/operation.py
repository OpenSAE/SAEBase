"""Budgeted SAE/Task smoke orchestration; Setup never owns a test.

Results remain in the existing Runtime-owned experiment leaf. Task workers are
isolated so native generation/training cannot silently exceed wall-time budget.
"""

from __future__ import annotations

import json
import os
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

    if not sites and not tasks:
        return []
    entries: list[dict[str, str | None]] = []
    model_key = str(cfg.model.name)
    model_repo = str(cfg.model.get("pretrained_name") or "")
    if "/" in model_repo:
        entries.append({"repo_id": model_repo, "revision": cfg.model.get("revision")})

    required_saes = [str(site["sae_key"]) for site in sites]
    configs = cfg.task.get("task_configs", {})
    for task_name in tasks:
        if task_name not in ("capture", "intervene", "generate"):
            continue
        own_config = configs.get(task_name, {})
        key = own_config.get("sae_key")
        if not key:
            defaults = list(cfg.model.get("default_saes", []))
            key = defaults[0] if defaults else None
        if key:
            required_saes.append(str(key))

    for sae_key in dict.fromkeys(required_saes):
        profile = get_profile(sae_key)
        if model_key not in profile.allowed_model_keys():
            raise ValueError(f"Selected SAE {sae_key!r} does not support Model {model_key!r}")
        repo_id = str(profile.repo_id or "")
        if "/" in repo_id:
            entries.append({"repo_id": repo_id, "revision": None})

    unique = {}
    for entry in entries:
        unique[(entry["repo_id"], entry["revision"])] = entry
    return list(unique.values())


def _run_worker(
    request: dict[str, Any], *, label: str, timeout: int, allow_network: bool
) -> dict[str, Any]:
    from saebase.runtime.experiment_paths import ARTIFACT_DIRS

    root = Path.cwd() / "analysis" / "precheck_jobs" / label
    for sub in ARTIFACT_DIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    inp = root / "metadata" / "request.json"
    out = root / "metadata" / "result.json"
    with inp.open("x", encoding="utf-8") as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
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
        proc = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return blocked(
            stage=label,
            code="BLOCKED_RESOURCE",
            reason=f"Real worker exceeded {timeout}s",
            attempted=["isolated worker started", "wall-time budget enforced"],
            repair="Provide enough compute or explicitly raise the bounded Precheck time budget.",
        )
    if not out.is_file():
        return {
            "status": "FAIL",
            "stage": label,
            "code": "WORKER_NO_RECEIPT",
            "reason": f"Worker exited {proc.returncode} without a result receipt; stderr: {proc.stderr[-500:]}",
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
    target = OmegaConf.merge(
        cfg,
        original,
        {
            "task": {
                "name": name,
                "execute": True,
                "device": str(cfg.task.get("device", "cpu")),
                "prompt": str(cfg.task.get("prompt", "")),
                "timeout_seconds": max_seconds,
            }
        },
        {"task": task},
    )
    if name == "generate":
        specified = int(target.task.get("max_new_tokens", max_tokens))
        if specified > max_tokens:
            raise ValueError("Task generation exceeds max_tokens budget")
        target.task.max_new_tokens = specified
    if name in ("circuits", "train", "explain"):
        if not bool(cfg.task.get("allow_native", False)):
            raise PermissionError(f"Native {name} requires task.allow_native=true")
        target.task.approved = True
    if name in ("benchmark", "circuits", "train", "explain") and not task:
        raise ValueError(
            f"Task {name} needs an explicit, bounded task.task_configs.{name} plan "
            "before a real smoke; Precheck must not run an unbounded default protocol"
        )
    if name == "train":
        plan = target.task.get("train_plan")
        if plan is None or int(plan.get("steps", max_seconds + 1)) > 2:
            raise ValueError(
                "Precheck Train requires a native TrainPlan bounded to at most two steps"
            )
    # Shipped Task YAMLs were originally Gemma-specific. For a different
    # Model, use its own default SAE and corresponding actual layer, unless
    # the caller explicitly selected both parameters for this Task.
    if name in ("capture", "intervene", "generate"):
        from saebase.api.sae_profiles import get_profile

        model_key = str(cfg.model.name)
        is_explicit_sae = "sae_key" in task
        is_explicit_layer = "layer" in task
        defaults = list(cfg.model.get("default_saes", []))
        if not is_explicit_sae:
            if not defaults:
                raise ValueError(f"Model {model_key!r} has no default SAE for {name}")
            target.task.sae_key = str(defaults[0])
        key = str(target.task.sae_key)
        profile = get_profile(key)
        if model_key not in profile.allowed_model_keys():
            raise ValueError(f"SAE {key!r} does not support model {model_key!r}")
        if not is_explicit_layer:
            target.task.layer = int(profile.layers[-1])
        elif int(target.task.layer) not in profile.layers:
            raise ValueError(f"Explicit layer {target.task.layer} is invalid for SAE {key!r}")
    return OmegaConf.to_container(target, resolve=True)


def run_precheck(cfg: DictConfig) -> EvaluationResult:
    mode = str(cfg.task.get("mode", "sae"))
    rows: list[dict[str, Any]] = []
    model_key = str(cfg.model.name)
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
        if bool(cfg.task.get("auto_setup", True)):
            try:
                from saebase.setup.assets import prepare_environment

                prepared = prepare_environment(
                    required_modules=(
                        ("huggingface_hub", "sae_lens", "transformer_lens", "transformers")
                        if (sites or any(t in ("capture", "intervene", "generate") for t in tasks))
                        else ("huggingface_hub",) if tasks else ()
                    ),
                    assets=_assets(cfg, sites, tasks),
                    allow_network=bool(cfg.task.get("allow_network", False)),
                    allow_install=bool(cfg.task.get("allow_install", False)),
                    allow_download=bool(cfg.task.get("allow_download", False)),
                    max_download_bytes=int(cfg.task.get("max_download_bytes", 0)),
                    install_timeout_seconds=max_seconds,
                )
                if prepared["status"] != "READY":
                    rows.append(prepared)
                    setup_ok = False
            except Exception as exc:
                rows.append(
                    error_result(exc, stage="setup", attempted=["Setup preparation requested"])
                )
                setup_ok = False
        allowed = bool(cfg.task.get("allow_compute", False))
        device = str(cfg.task.get("device", "cpu"))
        if device.startswith("cuda") and not bool(cfg.task.get("allow_gpu", False)):
            allowed = False
        if (sites or tasks) and not allowed:
            rows.append(
                blocked(
                    stage="authorization",
                    code="BLOCKED_AUTH",
                    reason="Real compute not authorized for this Precheck",
                    attempted=["read explicit execution permissions"],
                    repair="Set task.allow_compute=true and, for CUDA, task.allow_gpu=true, with an explicit resource budget.",
                )
            )
        if setup_ok and allowed:
            prompt = str(cfg.task.get("prompt", ""))
            for index, site in enumerate(sites):
                request = {
                    "kind": "sae",
                    "model_key": model_key,
                    "sae_key": site["sae_key"],
                    "layer": site["layer"],
                    "prompt": prompt,
                    "device": device,
                }
                rows.append(
                    _run_worker(
                        request,
                        label=f"sae_{index:02d}",
                        timeout=max_seconds,
                        allow_network=bool(cfg.task.get("allow_network", False)),
                    )
                )
            for index, name in enumerate(tasks):
                try:
                    task_cfg = _task_config(cfg, name, max_seconds, max_tokens)
                    request = {"kind": "task", "operation": name, "config": task_cfg}
                    rows.append(
                        _run_worker(
                            request,
                            label=f"task_{index:02d}_{name}",
                            timeout=max_seconds,
                            allow_network=bool(cfg.task.get("allow_network", False)),
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
