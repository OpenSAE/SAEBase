"""Task-owned automatic precheck policy for the public execution dispatcher.

Every executed task receives a persisted check receipt before its run method.
SAE operations validate their selected cohort and actual checkpoint; native
protocols prepare their inputs without rerunning the research task.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value

import os
from pathlib import Path
from tempfile import mkdtemp

from conf import load_task
from omegaconf import OmegaConf

from saebase.results import EvaluationResult


def _check_directory():
    root = Path.cwd() / "metadata" / "automatic_precheck"
    try:
        root.mkdir(parents=True)
    except FileExistsError:
        attempts = root / "attempts"
        attempts.mkdir(exist_ok=True)
        return Path(mkdtemp(prefix="retry_", dir=attempts))
    return root


def _failure(cfg, result, receipt):
    from saebase.runtime.experiment_paths import run_components

    return EvaluationResult(
        ok=False,
        task=str(cfg.task.name),
        evaluation=str(cfg.task.name),
        model_key=run_components(cfg)["model"],
        prompt_source=str(cfg.dataset.name),
        status="failed" if result.exit_code == 1 else "blocked",
        error="Automatic precheck did not pass; requested task was not executed",
        summary={"automatic_precheck": result.to_dict()},
        artifact_paths={"precheck": str(receipt)},
    ), str(receipt)


def record_load_failure(cfg, exc):
    """Record a failed shared resource load using the automatic precheck contract."""
    from saebase.runtime.experiment_results import write_json
    from tasks.precheck.report import error_result, summarize

    receipt = _check_directory() / "metadata" / "precheck_result.json"
    rows = [error_result(exc, stage="model_and_sae_load", attempted=["load model and SAE"])]
    summary = summarize(rows)
    write_json(
        receipt,
        {
            "schema": "saebase.precheck.v1",
            "mode": "performance",
            "model": str(cfg.model.name),
            "dataset": str(cfg.dataset.name),
            "rows": rows,
            "summary": summary,
        },
    )
    result = EvaluationResult(
        ok=False,
        task="precheck",
        evaluation="precheck",
        model_key=str(cfg.model.name),
        prompt_source=str(cfg.dataset.name),
        status="failed" if rows[0]["status"] == "FAIL" else "blocked",
        meta={"rows": rows},
    )
    return _failure(cfg, result, receipt)


def automatic_precheck(cfg):
    from tasks.precheck.task import PrecheckTask
    from saebase.extensions.registry import external_model
    from saebase.runtime.sae_operations import _ACTIVE_PAIR, resolve_sae

    name = str(cfg.task.name)
    check = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    check.task = load_task("precheck").task
    check.task.execute = True
    check.task.precheck.target_operation = name
    check.task.device = str(cfg.task.get("device", cfg.model.get("device", "auto")))
    check.task.extensions = list(cfg.task.get("extensions", []))
    check.task.prompt = str(cfg.task.get("prompt", "") or "")
    check.task.precheck.max_seconds = 300
    check.task.precheck.task_configs = {name: OmegaConf.to_container(cfg.task, resolve=True)}
    from saebase.runtime.performance import settings

    if (
        settings(cfg)["enabled"]
        and name in {"capture", "intervene", "generate"}
        and cfg.dataset.get("samples") is None
    ):
        check.task.precheck.mode = "performance"
        check.task.precheck.check_tasks = [name]
        check.task.performance = OmegaConf.to_container(cfg.task.performance, resolve=True)
        if name == "capture":
            check.task.precheck.capture_modes = [str(task_value(cfg.task, "mode", "whiten"))]
    if _ACTIVE_PAIR.get() is not None or (
        cfg.model.get("name")
        and external_model(str(cfg.model.name)) is not None
        and not check.task.extensions
    ):
        check.task.precheck.worker_mode = "in_process"
    if name in {"capture", "intervene", "generate"} and cfg.dataset.get("samples") is None:
        selected = resolve_sae(
            cfg.model, cfg.task.get("sae_key"), cfg.task.get("layer"), operation=name
        )
        check.task.precheck.sae_list = [{"sae": selected.sae_key, "layer": selected.layer}]
    else:
        check.task.precheck.mode = "task"
        check.task.precheck.check_tasks = [name]
        check.task.precheck.task_configs = {name: OmegaConf.to_container(cfg.task, resolve=True)}
    origin = Path.cwd()
    destination = _check_directory()
    try:
        os.chdir(destination)
        result = PrecheckTask().run(check)
    finally:
        os.chdir(origin)
    receipt = destination / "metadata" / "precheck_result.json"
    if result.ok:
        if check.task.precheck.mode == "performance":
            from saebase.runtime.performance import remember_recommendations

            remember_recommendations(result.summary.get("records", []))
        return None, str(receipt)
    return _failure(cfg, result, receipt)
