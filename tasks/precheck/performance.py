"""Real, bounded performance checks through the selected native task adapters.

Owns representative input selection, task-specific work units and reports.
Reference cohort sizes are reduced only inside clearly marked probe runs.
"""

from saebase.runtime.configuration.task_options import merge_task_overrides, task_owner, task_value

from pathlib import Path
import os
import time
import json
import traceback
from contextlib import ExitStack
from conf import load_task
from omegaconf import OmegaConf

from saebase.results import EvaluationResult
from saebase.runtime.performance import settings, session, stage
from saebase.runtime.experiment_results import write_json


def run_performance_precheck(cfg):
    if not cfg.task.get("execute", False):
        return EvaluationResult(
            ok=False,
            task="precheck",
            status="configured_only",
            evaluation="performance",
            model_key=str(cfg.model.name),
            prompt_source=str(cfg.dataset.name),
            summary={
                "status": "NOT_RUN",
                "reason": "Set task.execute=true to run real performance probes",
            },
        )
    from saebase.registry import get_task_plugin
    from saebase.runtime.sae_operations import resolve_sae, load_selection, loaded_pair_session
    from models.numerics import configure_task

    configure_task(cfg)

    options = settings(cfg)
    targets = list(task_value(cfg.task, "check_tasks") or ["capture", "intervene", "generate"])
    root = Path.cwd()
    records = []
    started = time.perf_counter()
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    overrides = task_value(cfg.task, "task_configs", {})
    pair = None
    with ExitStack() as resources:
        for target in targets:
            modes = (
                list(task_value(cfg.task, "capture_modes", ["whiten", "non_whiten", "g_s", "I"]))
                if target == "capture"
                else [None]
            )
            for mode in modes:
                name = target + ("_" + mode if mode else "")
                child = OmegaConf.create(raw_cfg)
                task_override = overrides.get(target, {})
                if target == "capture":
                    from tasks.capture.configuration import migrate_capture_task

                    task_override = migrate_capture_task(task_override)
                child.task = merge_task_overrides(
                    load_task(target).task,
                    task_override,
                )
                OmegaConf.set_struct(child, False)
                child.task.performance = OmegaConf.to_container(cfg.task.performance, resolve=True)
                child.task.device = str(cfg.task.get("device", "auto"))
                child.task.performance.enabled = True
                child.task.execute = True
                if mode is not None:
                    task_owner(child.task).mode = mode
                sae_list = task_value(cfg.task, "sae_list")
                if sae_list not in (None, "default", "all"):
                    entry = sae_list if isinstance(sae_list, str) else sae_list[0]
                    child.task.sae_key = (
                        entry.get("sae", entry.get("key")) if hasattr(entry, "get") else entry
                    )
                    if hasattr(entry, "get") and entry.get("layer") is not None:
                        child.task.layer = entry.layer
                explicit_samples = task_value(child.task, "sample_ids")
                if explicit_samples is not None:
                    explicit_samples = list(explicit_samples)[: options["probe_samples"]]
                    task_owner(child.task).sample_ids = explicit_samples
                if explicit_samples is None:
                    child.dataset.sample_count = options["probe_samples"]
                    if target == "capture":
                        child.task.capture.selection.cohort_filter.reference_sample_count = options[
                            "probe_samples"
                        ]
                if target in {"intervene", "generate"}:
                    task_owner(child.task).feature_selection = "capture_top"
                    task_owner(child.task).inline_results = False
                folder = root / "tmp/performance" / name
                folder.mkdir(parents=True, exist_ok=True)
                for directory in ["reports", "metadata", "logs", "tmp", "figures"]:
                    (folder / directory).mkdir(exist_ok=True)
                record = {
                    "task": name,
                    "probe_questions_requested": len(explicit_samples)
                    if explicit_samples is not None
                    else options["probe_samples"],
                    "scientific_acceptance": False,
                    "cohort_scope": "bounded performance probe; not the production research cohort",
                }
                try:
                    with session(child):
                        with stage("model_and_sae_load"):
                            selected = resolve_sae(
                                child.model,
                                child.task.get("sae_key"),
                                child.task.get("layer"),
                                operation=target,
                            )
                            if pair is None:
                                pair = load_selection(selected, str(cfg.task.get("device", "auto")))
                                resources.enter_context(loaded_pair_session(pair))
                            elif (
                                pair.identity.model_key,
                                pair.identity.sae_key,
                                pair.identity.layer,
                            ) != (selected.model_key, selected.sae_key, selected.layer):
                                raise ValueError(
                                    "One performance Precheck must select the same model/SAE/site for all target tasks"
                                )
                        plugin = get_task_plugin(target).task_class()
                        plugin.validate_cfg(child)
                        os.chdir(folder)
                        # A task-owned probe uses the same scientific operation, not
                        # a second Runtime/automatic-precheck recursion.
                        OmegaConf.save(child, folder / "metadata/config.yaml")
                        with loaded_pair_session(pair), stage("task_execution"):
                            result = plugin.run_validated(child)
                        if mode == "I" and options.get("require_batch_invariance"):
                            from tasks.capture.performance import verify_batch_artifact

                            artifacts = sorted((folder / "reports/I").glob("*.npz"))
                            if artifacts:
                                with stage("full_state_batch_check"):
                                    record["full_state_batch_check"] = verify_batch_artifact(
                                        pair, artifacts[0], batch_size=16
                                    )
                        record.update(
                            status="PASS"
                            if result.status == "executed"
                            and not result.completion.get("counts", {}).get("failed")
                            and not result.completion.get("counts", {}).get("unsupported")
                            and not result.completion.get("missing")
                            else "FAIL",
                            result_status=result.status,
                            completion=result.completion,
                            identity=pair.identity.to_dict(),
                            diagnostics=result.diagnostics,
                        )
                        if record["status"] == "FAIL":
                            record["error"] = (
                                "; ".join(
                                    str(row.get("reason", ""))
                                    for row in result.completion.get("reasons", [])[:3]
                                )
                                or result.error
                                or "Task probe did not complete its requested numerical work"
                            )
                        write_json(folder / "metadata/probe_result.json", result.to_dict())
                        os.chdir(root)
                    perf = json.loads((root / "reports/performance.json").read_text())
                    # session was created at the outer precheck directory. Keep a
                    # distinct copy for every task and the original raw observations.
                    write_json(folder / "reports/performance.json", perf)
                    record.update(
                        per_sample=perf["per_sample"],
                        question_excluding_tuning=perf["question_excluding_tuning"],
                        phases=perf["phases"],
                        optimizations=perf["optimization_trials"],
                        report=str(folder / "reports/performance.json"),
                    )
                except Exception as exc:
                    record.update(
                        status="FAIL",
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                finally:
                    os.chdir(root)
                records.append(record)
                write_json(
                    root / "reports/performance_precheck.json",
                    {"status": "RUNNING", "records": records},
                )
        with stage("checkpoint_recheck"):
            resources.close()
    ok = bool(records) and all(row["status"] == "PASS" for row in records)
    payload = {
        "schema": "saebase.performance_precheck.v1",
        "status": "PASS" if ok else "FAIL",
        "seconds": time.perf_counter() - started,
        "records": records,
        "selection": "Dataset fixed sampling seed; probe count explicitly reduced; report distribution is descriptive",
        "production_batches_started": False,
    }
    write_json(root / "reports/performance_precheck.json", payload)
    write_json(
        root / "metadata/precheck_result.json",
        {
            "schema": "saebase.precheck.v1",
            "mode": "performance",
            "model": str(cfg.model.name),
            "dataset": str(cfg.dataset.name),
            "summary": {"status": payload["status"]},
            "rows": records,
            "performance_report": "reports/performance_precheck.json",
        },
    )
    lines = [
        "# Performance precheck report",
        "",
        f"Status:{payload['status']};Total time-consuming:{payload['seconds']:.2f} seconds.",
        "",
        "The following are real small-scale probes, not batch results..",
        "",
        "| Tasks | Status | Units | Number | Average seconds | P50 | P95 | Max. sec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        for unit, values in row.get("per_sample", {}).items():
            lines.append(
                f"| {row['task']} | {row['status']} | {unit} | {values['count']} | {values['mean_seconds']:.4f} | {values['p50_seconds']:.4f} | {values['p95_seconds']:.4f} | {values['max_seconds']:.4f} |"
            )
    lines += [
        "",
        "## Less time spent on each topic after the first modulation",
        "",
        "Keeps each issue of cache preparation, numerical control and writing; shares the load and reference statistics for the record of the reference phase. The fractions of a small number of probes describe only this measurement.",
        "",
        "| Tasks | Number | Average seconds | P50 | P95 | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        values = row.get("question_excluding_tuning", {})
        if not values.get("count"):
            values = row.get("per_sample", {}).get("question", {})
        if values.get("count"):
            lines.append(
                f"| {row['task']} | {values['count']} | {values['mean_seconds']:.4f} | {values['p50_seconds']:.4f} | {values['p95_seconds']:.4f} | {values['min_seconds']:.4f} | {values['max_seconds']:.4f} |"
            )
    lines += [
        "",
        "## Optimizing benefits",
        "",
        "| Tasks | Optimization | Select batch | Measured acceleration multiplier | Numerical Check |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in records:
        for trial in row.get("optimizations", []):
            if "speedup" in trial:
                optimization = trial.get("optimization", trial.get("optimizations", []))
                lines.append(
                    f"| {row['task']} | {optimization} | {trial.get('selected_batch_size')} | {trial['speedup']:.3f} | {'Adopted and adopted' if trial.get('selected') else 'No use for faster candidates, keep the original configuration'} |"
                )
        if row.get("error"):
            lines += ["", row["task"] + ":" + row["error"]]
    lines += [
        "",
        "## Phase time-consuming",
        "",
        "| Tasks | Phase | Number | Total seconds | Average seconds | P95 seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        for name, phase in row.get("phases", {}).items():
            lines.append(
                f"| {row['task']} | {name} | {phase['count']} | {phase['total_seconds']:.4f} | {phase['mean_seconds']:.4f} | {phase['p95_seconds']:.4f} |"
            )
    lines += [
        "",
        "## Mitigating costs",
        "",
        "| Tasks | Tunnel total seconds | Benchmark seconds/Units | Select seconds/Units | Expected return to unit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        for trial in row.get("optimizations", []):
            if "baseline_seconds" in trial:
                base = trial["baseline_seconds"] / trial["units"]
                chosen = base / trial["speedup"]
                units = trial.get("break_even_units")
                lines.append(
                    f"| {row['task']} | {trial.get('total_tuning_seconds', trial['search_seconds']):.3f} | {base:.5f} | {chosen:.5f} | {round(units, 1) if units is not None else 'No measured gain'} |"
                )
    lines += [
        "",
        "Accelerator multiple comparison of the same input, reference achievement for the same workload and candidate; Cache preparation, search and complete sample time; see also JSON.Batch change of the float result will reject the candidate; OOM Candidatures will be recorded and stopped for larger. batch Search.",
        "",
        "Complete phase time-consuming, P90/P99,Sample-by-sampling data, speed per candidate/Visibility/Both the error and the cause of the failure are stored in performance_precheck.json.",
    ]
    (root / "reports/performance_precheck.md").write_text("\n".join(lines) + "\n")
    return EvaluationResult(
        ok=ok,
        task="precheck",
        status="executed" if ok else "failed",
        evaluation="performance",
        model_key=str(cfg.model.name),
        prompt_source=str(cfg.dataset.name),
        summary=payload,
        artifact_paths={
            "performance": str(root / "reports/performance_precheck.json"),
            "performance_report": str(root / "reports/performance_precheck.md"),
        },
        error=None if ok else "Some performance probes failed; see recorded task-specific errors",
    )
