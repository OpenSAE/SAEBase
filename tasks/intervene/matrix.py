"""Execute and monitor the primary Top N intervention/generation matrix.

Uses the public runtime lifecycle, model-specific interpreters, verified saved
capture rankings and task-owned resume. One GPU worker runs at a time; a failed
cell/model cannot prevent independent peers from running. Status is persisted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from omegaconf import OmegaConf
from conf import load_group
from saebase.runtime.experiment_results import write_json as save


class FeatureTaskMatrix:
    def __init__(self, config, root, report):
        self.config = Path(config).resolve()
        self.root = Path(root).resolve()
        self.report = Path(report).resolve()
        self.plan = OmegaConf.load(self.config).matrix
        self.tag = str(self.plan.tag)
        self.report.mkdir(parents=True, exist_ok=True)
        (self.report / "cells").mkdir(exist_ok=True)
        (self.report / "inputs").mkdir(exist_ok=True)

    def cfg(self, model, dataset, operation):
        from hydra import compose
        from models.loading_policy import resolve_config_devices

        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model.model}",
                f"dataset={dataset}",
                f"task={operation}",
                f"paths.root={self.root}",
                "task.execute=true",
                f"task.sae_key={model.sae_key}",
                f"task.layer={model.layer}",
                f"experiment.sub_name={self.tag}",
                f"seed={self.plan.get('seed', 42)}",
            ],
        )
        if operation == "capture":
            cfg.task.mode = "whiten"
            cfg.task.neuronpedia.enabled = False
            cfg.task.neuronpedia.allow_network = False
        else:
            cfg.task.feature_selection = "capture_top"
            cfg.task.feature_ranking = "whiten"
            cfg.task.top_feature_num = 10
            cfg.task.value = 0.0
            cfg.task.inline_results = False
            cfg.task.continue_on_error = True
            cfg.task.reuse_prompt_observations = True
            if operation == "generate":
                cfg.task.intervention = "ablate"
                cfg.task.max_new_tokens = int(self.plan.get("max_new_tokens", 8))
                cfg.task.temperature = 0.0
        resolve_config_devices(cfg)
        return cfg

    def cell_path(self, name, key, task):
        return self.report / "cells" / f"{name}--{key}--{task}.json"

    def leaf(self, name, key, task, tag=None):
        return (
            self.root
            / "experiment"
            / name
            / key
            / task
            / f"{tag or self.tag}_{self.plan.get('seed', 42)}"
        )

    def freeze_inputs(self, cfg, pair):
        from saebase.datasets.selection import select_dataset

        dest = self.report / "inputs" / f"{cfg.dataset.name}--{cfg.model.name}.jsonl"
        if not dest.exists():
            selection = select_dataset(
                cfg.dataset,
                model_key=pair.identity.model_key,
                model=pair.model.model,
                loaded_model_name=pair.identity.loaded_model_name,
            )
            rows = list(selection)
            dest.write_text(
                "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
            )
            save(dest.with_suffix(".source.json"), selection.metadata)
        rows = [json.loads(line) for line in dest.read_text().splitlines()]
        if not rows:
            raise ValueError("Empty frozen Dataset cohort")
        cfg.dataset.prompt_file = str(dest)
        return rows

    def compatible_source(self, folder, rows, pair, hashes):
        ledger = folder / "reports/capture_samples.jsonl"
        if not ledger.is_file():
            return False
        found = set()
        for line in ledger.open():
            result = json.loads(line).get("result", {})
            identity = result.get("input_identity") or {}
            actual = (
                result.get("model_key"),
                result.get("sae_key"),
                result.get("layer"),
                result.get("site"),
            )
            expected = (
                pair.identity.model_key,
                pair.identity.sae_key,
                pair.site.layer,
                pair.site.hook_name,
            )
            if (
                actual == expected
                and result.get("ranking") == "absolute_zscore"
                and identity.get("model_sha256") == hashes[0]
                and identity.get("sae_sha256") == hashes[1]
                and len(result.get("top_features", [])) >= 10
            ):
                found.add(result["prompt_sha256"])
        return all(hashlib.sha256(r["prompt"].encode()).hexdigest() in found for r in rows)

    def source(self, model, alias, cfg, pair, hashes, rows):
        candidates = [
            self.leaf(cfg.dataset.name, model.model, "capture", self.tag + "_selection"),
            self.leaf(cfg.dataset.name, model.model, "capture", "whiten_matrix_20261114"),
        ]
        for folder in candidates:
            if self.compatible_source(folder, rows, pair, hashes):
                return folder
        capture_cfg = self.cfg(model, alias, "capture")
        capture_cfg.dataset = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
        capture_cfg.experiment.sub_name = self.tag + "_selection"
        folder = candidates[0]
        if (folder / "metadata/manifest.json").exists():
            index = 1
            while folder.with_name(f"{self.tag}_selection_attempt{index}_42").exists():
                index += 1
            folder = folder.with_name(f"{self.tag}_selection_attempt{index}_42")
            capture_cfg.experiment.sub_name = folder.name.rsplit("_", 1)[0]
        self.execute(capture_cfg, folder)
        if not self.compatible_source(folder, rows, pair, hashes):
            raise RuntimeError(f"Capture did not produce all requested Top 10 rankings: {folder}")
        return folder

    def execute(self, cfg, folder):
        from saebase.runtime.train import execute_run

        origin = Path.cwd()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(folder)
            return execute_run(cfg, folder)
        finally:
            os.chdir(origin)

    def worker(self, key, task):
        from hydra import initialize_config_module
        from tasks.sae_runtime import (
            loaded_pair_session,
            resolve_sae,
            load_selection,
            pair_state_fingerprints,
        )
        from saebase.runtime.experiment_results import inspect_run
        import torch

        model = next(r for r in self.plan.models if r.model == key)
        with initialize_config_module(version_base=None, config_module="conf"):
            cfg = self.cfg(model, self.plan.datasets[0], task)
            try:
                pair = load_selection(
                    resolve_sae(cfg.model, model.sae_key, model.layer, operation=task), "cuda"
                )
            except Exception as exc:
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                save(self.report / f"load-{key}-{task}.json", error)
                for alias in self.plan.datasets:
                    name = str(load_group("dataset", str(alias)).dataset.name)
                    path = self.cell_path(name, key, task)
                    if (
                        not path.exists()
                        or json.loads(path.read_text()).get("status") != "complete"
                    ):
                        save(
                            path,
                            {
                                "dataset": name,
                                "model": key,
                                "task": task,
                                "status": "blocked",
                                "error": error,
                            },
                        )
                raise
            with loaded_pair_session(pair):
                hashes = pair_state_fingerprints(pair)
                save(
                    self.report / f"pair-{key}.json",
                    {
                        "identity": pair.identity.to_dict(),
                        "model_sha256": hashes[0],
                        "sae_sha256": hashes[1],
                    },
                )
                for alias in self.plan.datasets:
                    cfg = self.cfg(model, alias, task)
                    name = str(cfg.dataset.name)
                    record = self.cell_path(name, key, task)
                    folder = self.leaf(name, key, task)
                    if record.exists():
                        old = json.loads(record.read_text())
                        if old.get("status") == "complete":
                            continue
                        folder = Path(old.get("run_dir", folder))
                        cfg.experiment.sub_name = folder.name.rsplit("_", 1)[0]
                    cell = {
                        "dataset": name,
                        "model": key,
                        "task": task,
                        "layer": int(model.layer),
                        "run_dir": str(folder),
                        "status": "running",
                        "started_at": time.time(),
                    }
                    save(record, cell)
                    try:
                        rows = self.freeze_inputs(cfg, pair)
                        cell["selected_questions"] = len(rows)
                        cell["requested_measurements"] = len(rows) * 10
                        source = self.source(model, alias, cfg, pair, hashes, rows)
                        cfg.task.capture_source = str(source)
                        cell["capture_source"] = str(source)
                        if (folder / "metadata/manifest.json").exists():
                            state = inspect_run(folder, expected_cfg=cfg)
                            if state["action"] == "skip":
                                save(record, {**cell, "status": "complete"})
                                continue
                            cfg.runtime.mode = "resume"
                        save(record, cell)
                        result = self.execute(cfg, folder)
                        cell["scientific_summary"] = self.scientific_summary(folder)
                        cell.update(
                            status="complete" if result.ok else "incomplete",
                            completion=result.completion,
                            diagnostics=result.diagnostics,
                        )
                    except Exception as exc:
                        cell.update(
                            status="failed",
                            error={
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                    cell["finished_at"] = time.time()
                    save(record, cell)
                    print(
                        json.dumps(
                            {
                                k: cell.get(k)
                                for k in [
                                    "dataset",
                                    "model",
                                    "task",
                                    "status",
                                    "completion",
                                    "error",
                                ]
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    torch.cuda.empty_cache()

    def scientific_summary(self, folder):
        from tasks.intervene.top_features import read_feature_results

        rows = read_feature_results(folder)
        completed = [row for row in rows if row.get("status", "completed") == "completed"]
        task = json.loads((folder / "metadata/manifest.json").read_text())["task"]
        stats = {"completed": len(completed), "failed": len(rows) - len(completed)}
        if task == "intervene":
            stats["top1_changed"] = sum(
                row["result"]["top_baseline_token_ids"][0]
                != row["result"]["top_edited_token_ids"][0]
                for row in completed
            )
            stats["nonzero_logit_effect"] = sum(
                row["result"]["max_abs_logit_change"] > 0 for row in completed
            )
            stats["mean_max_abs_logit_change"] = sum(
                row["result"]["max_abs_logit_change"] for row in completed
            ) / max(1, len(completed))
        else:
            runs = [row["result"]["result"]["runs"][0] for row in completed]
            stats["changed_continuations"] = sum(not run["same_continuation"] for run in runs)
            stats["context_limited"] = sum(
                run.get("edited_stop_reason") == "context_limit" for run in runs
            )
            stats["generated_tokens"] = sum(len(run["edited_generated_token_ids"]) for run in runs)
        return stats

    def summary(self, active=None):
        cells = []
        totals = {
            t: {"completed": 0, "failed_measurements": 0, "requested": 0, "complete_cells": 0}
            for t in self.plan.tasks
        }
        for task in self.plan.tasks:
            for model in self.plan.models:
                for alias in self.plan.datasets:
                    name = str(load_group("dataset", str(alias)).dataset.name)
                    path = self.cell_path(name, model.model, task)
                    cell = (
                        json.loads(path.read_text())
                        if path.exists()
                        else {
                            "dataset": name,
                            "model": str(model.model),
                            "task": str(task),
                            "status": "pending",
                        }
                    )
                    folder = Path(cell.get("run_dir", self.leaf(name, model.model, task)))
                    progress = folder / "metadata" / f"{task}_progress.json"
                    if progress.exists():
                        cell["progress"] = json.loads(progress.read_text())
                    stats = cell.get("progress", {})
                    total = totals[task]
                    total["completed"] += stats.get(
                        "completed", cell.get("completion", {}).get("completed", 0)
                    )
                    total["failed_measurements"] += stats.get("failed", 0)
                    total["requested"] += cell.get("requested_measurements", 0)
                    total["complete_cells"] += cell.get("status") == "complete"
                    cells.append(cell)
        result = {
            "updated_at": time.time(),
            "active": active,
            "expected_cells": len(self.plan.tasks)
            * len(self.plan.models)
            * len(self.plan.datasets),
            "totals": totals,
            "cells": cells,
        }
        save(self.report / "summary.json", result)
        lines = [
            "# 主要探究配置：Intervene / Generate 进度",
            "",
            "配置：whiten Top 10，逐特征清零，generate 贪心生成8 token；10模型×12数据集×2任务。",
            "",
            "当前进程：" + str(active),
            "",
            "| 任务 | 已完成特征实验 | 完成数据集×模型单元 | 当前失败特征实验 |",
            "| --- | --- | --- | --- |",
        ]
        for task, total in totals.items():
            lines.append(
                f"| {task} | {total['completed']} | {total['complete_cells']}/120 | {total['failed_measurements']} |"
            )
        lines += [
            "",
            "| 模型 | 数据集 | 任务 | 状态 | 已完成/计划特征数 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cell in cells:
            if cell["status"] != "pending":
                stat = cell.get("progress", {})
                lines.append(
                    f"| {cell['model']} | {cell['dataset']} | {cell['task']} | {cell['status']} | {stat.get('completed', 0)}/{cell.get('requested_measurements', '待准备')} |"
                )
        (self.report / "progress_zh.md").write_text("\n".join(lines) + "\n")
        return result

    def run(self):
        with (self.report / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env_config = OmegaConf.load(self.root / "conf/runtime/environments.yaml")
            for attempt in range(int(self.plan.get("max_passes", 2))):
                for task in self.plan.tasks:
                    for model in self.plan.models:
                        snapshot = self.summary()
                        model_cells = [
                            c
                            for c in snapshot["cells"]
                            if c["model"] == model.model and c["task"] == task
                        ]
                        if all(c["status"] == "complete" for c in model_cells):
                            continue
                        python = str(
                            env_config.get("models", {})
                            .get(model.model, {})
                            .get("python", env_config.default.python)
                        )
                        command = [
                            python,
                            "-m",
                            "tasks.intervene.matrix",
                            "--config",
                            str(self.config),
                            "--root",
                            str(self.root),
                            "--report",
                            str(self.report),
                            "--worker",
                            str(model.model),
                            "--task",
                            str(task),
                        ]
                        log = self.report / f"{task}-{model.model}-pass{attempt + 1}.log"
                        with log.open("a") as stream:
                            process = subprocess.Popen(
                                command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT
                            )
                            active = {
                                "model": str(model.model),
                                "task": str(task),
                                "pid": process.pid,
                                "pass": attempt + 1,
                                "log": str(log),
                            }
                            save(
                                self.report / "active_process.json", {**active, "command": command}
                            )
                            while process.poll() is None:
                                self.summary(active)
                                time.sleep(15)
                        save(
                            self.report / f"exit-{task}-{model.model}-pass{attempt + 1}.json",
                            {"exit_code": process.returncode, "command": command},
                        )
            summary = self.summary()
            done = all(c["status"] == "complete" for c in summary["cells"])
            save(self.report / "finished.json", {"complete": done, "finished_at": time.time()})
            return 0 if done else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/experiment/feature_tasks_matrix.yaml")
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--report", required=True)
    parser.add_argument("--worker")
    parser.add_argument("--task", choices=["intervene", "generate"])
    args = parser.parse_args()
    matrix = FeatureTaskMatrix(args.config, args.root, args.report)
    if args.worker:
        matrix.worker(args.worker, args.task)
        return 0
    return matrix.run()


if __name__ == "__main__":
    raise SystemExit(main())
