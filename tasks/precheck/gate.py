"""Task-owned automatic precheck policy for the public execution dispatcher.

Every executed task receives a persisted check receipt before its run method.
SAE operations validate their selected cohort and actual checkpoint; native
protocols prepare their inputs without rerunning the research task.
"""

from __future__ import annotations

from importlib.resources import files
import os
from pathlib import Path

from omegaconf import OmegaConf

from saebase.results import EvaluationResult


def automatic_precheck(cfg):
    from tasks.precheck.task import PrecheckTask
    from saebase.extensions.registry import external_model
    from saebase.runtime.experiment_paths import run_components
    from tasks.sae_runtime import _ACTIVE_PAIR, resolve_sae

    name = str(cfg.task.name)
    check = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    check.task = OmegaConf.load(files("conf") / "task" / "precheck.yaml").task
    check.task.execute = True
    check.task.target_operation = name
    check.task.device = str(cfg.task.get("device", cfg.model.get("device", "auto")))
    check.task.extensions = list(cfg.task.get("extensions", []))
    check.task.prompt = str(cfg.task.get("prompt", "") or "")
    check.task.max_seconds = 300
    check.task.task_configs = {name: OmegaConf.to_container(cfg.task, resolve=True)}
    from saebase.runtime.performance import settings

    if (
        settings(cfg)["enabled"]
        and name in {"capture", "intervene", "generate"}
        and cfg.dataset.get("samples") is None
    ):
        check.task.mode = "performance"
        check.task.check_tasks = [name]
        check.task.performance = OmegaConf.to_container(cfg.task.performance, resolve=True)
        if name == "capture":
            check.task.capture_modes = [str(cfg.task.get("mode", "whiten"))]
    if _ACTIVE_PAIR.get() is not None or (
        cfg.model.get("name")
        and external_model(str(cfg.model.name)) is not None
        and not check.task.extensions
    ):
        check.task.worker_mode = "in_process"
    if name in {"capture", "intervene", "generate"} and cfg.dataset.get("samples") is None:
        selected = resolve_sae(
            cfg.model, cfg.task.get("sae_key"), cfg.task.get("layer"), operation=name
        )
        check.task.sae_list = [{"sae": selected.sae_key, "layer": selected.layer}]
    else:
        check.task.mode = "task"
        check.task.check_tasks = [name]
        check.task.task_configs = {name: OmegaConf.to_container(cfg.task, resolve=True)}
    origin = Path.cwd()
    destination = origin / "tmp" / "automatic_precheck"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(destination)
        result = PrecheckTask().run(check)
    finally:
        os.chdir(origin)
    receipt = destination / "metadata" / "precheck_result.json"
    if result.ok:
        if check.task.mode == "performance":
            from saebase.runtime.performance import remember_recommendations

            remember_recommendations(result.summary.get("records", []))
        return None, str(receipt)
    return EvaluationResult(
        ok=False,
        task=name,
        evaluation=name,
        model_key=run_components(cfg)["model"],
        prompt_source=str(cfg.dataset.name),
        status="failed" if result.exit_code == 1 else "blocked",
        error="Automatic precheck did not pass; requested task was not executed",
        summary={"automatic_precheck": result.to_dict()},
        artifact_paths={"precheck": str(receipt)},
    ), str(receipt)
