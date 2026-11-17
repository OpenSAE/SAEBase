"""Real, bounded performance checks through the selected native task adapters.

Owns representative input selection, task-specific work units and reports.
Reference cohort sizes are reduced only inside clearly marked probe runs.
"""

from pathlib import Path
import os
import time
import json
import traceback
from contextlib import ExitStack
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
    from tasks.sae_runtime import resolve_sae, load_selection, loaded_pair_session
    from models.numerics import configure_task

    configure_task(cfg)

    options = settings(cfg)
    targets = list(cfg.task.get("check_tasks") or ["capture", "intervene", "generate"])
    root = Path.cwd()
    records = []
    started = time.perf_counter()
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    overrides = cfg.task.get("task_configs", {})
    pair = None
    with ExitStack() as resources:
        for target in targets:
            modes = (
                list(cfg.task.get("capture_modes", ["whiten", "non_whiten", "g_s", "I"]))
                if target == "capture"
                else [None]
            )
            for mode in modes:
                name = target + ("_" + mode if mode else "")
                child = OmegaConf.create(raw_cfg)
                child.task = OmegaConf.merge(
                    OmegaConf.load(Path(__file__).parents[2] / "conf/task" / f"{target}.yaml").task,
                    overrides.get(target, {}),
                )
                OmegaConf.set_struct(child, False)
                child.task.performance = OmegaConf.to_container(cfg.task.performance, resolve=True)
                child.task.device = str(cfg.task.get("device", "auto"))
                child.task.performance.enabled = True
                child.task.execute = True
                if mode is not None:
                    child.task.mode = mode
                if cfg.task.get("sae_list") not in (None, "default", "all"):
                    entry = (
                        cfg.task.sae_list
                        if isinstance(cfg.task.sae_list, str)
                        else cfg.task.sae_list[0]
                    )
                    child.task.sae_key = (
                        entry.get("sae", entry.get("key")) if hasattr(entry, "get") else entry
                    )
                    if hasattr(entry, "get") and entry.get("layer") is not None:
                        child.task.layer = entry.layer
                child.dataset.sample_count = options["probe_samples"]
                if "sar" in child.task:
                    child.task.sar.cohort_filter.reference_sample_count = options["probe_samples"]
                if target in {"intervene", "generate"}:
                    child.task.feature_selection = "capture_top"
                    child.task.inline_results = False
                folder = root / "tmp/performance" / name
                folder.mkdir(parents=True, exist_ok=True)
                for directory in ["reports", "metadata", "logs", "tmp", "figures"]:
                    (folder / directory).mkdir(exist_ok=True)
                record = {
                    "task": name,
                    "probe_questions_requested": options["probe_samples"],
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
        "# 性能预检报告",
        "",
        f"状态：{payload['status']}；总耗时：{payload['seconds']:.2f} 秒。",
        "",
        "以下为真实小规模探针，不是生产批次结果。每题计时包括调优开销；共享预处理与阶段耗时单独列出。",
        "",
        "| 任务 | 状态 | 单位 | 数量 | 平均秒 | P50 | P95 | 最大秒 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        for unit, values in row.get("per_sample", {}).items():
            lines.append(
                f"| {row['task']} | {row['status']} | {unit} | {values['count']} | {values['mean_seconds']:.4f} | {values['p50_seconds']:.4f} | {values['p95_seconds']:.4f} | {values['max_seconds']:.4f} |"
            )
    lines += [
        "",
        "## 扣除首次调优后的每题耗时",
        "",
        "保留每题缓存准备、数值控制和写入；共享加载与参考统计见阶段记录。少量探针的分位数仅描述本次测量。",
        "",
        "| 任务 | 数量 | 平均秒 | P50 | P95 | 最小 | 最大 |",
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
        "## 实测优化收益",
        "",
        "| 任务 | 优化 | 选中batch | 实测加速倍数 | 数值检查 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in records:
        for trial in row.get("optimizations", []):
            if "speedup" in trial:
                optimization = trial.get("optimization", trial.get("optimizations", []))
                lines.append(
                    f"| {row['task']} | {optimization} | {trial.get('selected_batch_size')} | {trial['speedup']:.3f} | {'通过并采用' if trial.get('selected') else '无可采用的更快候选，保留原配置'} |"
                )
        if row.get("error"):
            lines += ["", row["task"] + "：" + row["error"]]
    lines += [
        "",
        "## 阶段耗时",
        "",
        "| 任务 | 阶段 | 次数 | 总秒 | 平均秒 | P95秒 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        for name, phase in row.get("phases", {}).items():
            lines.append(
                f"| {row['task']} | {name} | {phase['count']} | {phase['total_seconds']:.4f} | {phase['mean_seconds']:.4f} | {phase['p95_seconds']:.4f} |"
            )
    lines += [
        "",
        "## 调优成本",
        "",
        "| 任务 | 调优总秒 | 基准秒/单位 | 选中秒/单位 | 预计回本单位数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        for trial in row.get("optimizations", []):
            if "baseline_seconds" in trial:
                base = trial["baseline_seconds"] / trial["units"]
                chosen = base / trial["speedup"]
                units = trial.get("break_even_units")
                lines.append(
                    f"| {row['task']} | {trial.get('total_tuning_seconds', trial['search_seconds']):.3f} | {base:.5f} | {chosen:.5f} | {round(units, 1) if units is not None else '无实测收益'} |"
                )
    lines += [
        "",
        "加速倍数比较相同输入、相同工作量的参考实现与选中候选；缓存准备、搜索和完整样本耗时另见JSON。批量改变浮点结果时会拒绝该候选；OOM候选会记录并停止向更大batch搜索。",
        "",
        "完整的阶段耗时、P90/P99、逐样本数据、每个候选的速度/显存/误差及失败原因均保存在performance_precheck.json。",
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
