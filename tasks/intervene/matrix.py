"""Execute and monitor the primary Top N intervention/generation matrix.

Uses the public runtime lifecycle, model-specific interpreters, verified saved
capture rankings and task-owned resume. One GPU worker runs at a time; a failed
cell/model cannot prevent independent peers from running. Status is persisted.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_owner

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
    def __init__(self, config, root, report=None):
        self.config = Path(config).resolve()
        self.root = Path(root).resolve()
        from saebase.runtime.experiment_paths import experiment_name, matrix_report

        self.plan = OmegaConf.load(self.config).matrix
        self.tag = experiment_name(self.plan.tag)
        self.selection_name = "feature_selection"
        self.report = (
            Path(report).resolve()
            if report
            else matrix_report(self.root, "intervene", self.tag)
        )
        self.ranking = str(self.plan.get("feature_ranking", "whiten"))
        self.count = self.plan.get("feature_count", 10)
        if self.ranking not in {"whiten", "non_whiten"} or type(self.count) is not int or self.count < 1:
            raise ValueError("Feature matrix requires a supported ranking and positive feature_count")
        self.intervention_value = float(self.plan.get("intervention_value", 0.0))
        generation = self.plan.get("generation", {})
        self.generation = {
            "intervention": generation.get("intervention", "ablate"),
            "max_new_tokens": generation.get("max_new_tokens", self.plan.get("max_new_tokens", 8)),
            "temperature": generation.get("temperature", 0.0),
        }
        # Preserve the previous source candidate for older matrix recipes.
        self.capture_source_tag = self.plan.get("capture_source_tag", "cohort_ranking")
        self.report.mkdir(parents=True, exist_ok=True)
        save(self.report / "scope.json", {
            "datasets": [str(load_group("dataset", alias).dataset.name) for alias in self.plan.datasets],
            "models": [row.model for row in self.plan.models],
            "tasks": list(self.plan.get("tasks", ["intervene", "generate"])),
        })
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
            task_owner(cfg.task).mode = self.ranking
            cfg.task.capture.selection.candidate_count = self.count
            cfg.task.capture.output.feature_count = self.count
            task_owner(cfg.task).neuronpedia.enabled = False
            task_owner(cfg.task).neuronpedia.allow_network = False
        else:
            task_owner(cfg.task).feature_selection = "capture_top"
            task_owner(cfg.task).feature_ranking = self.ranking
            task_owner(cfg.task).top_feature_num = self.count
            task_owner(cfg.task).value = self.intervention_value
            task_owner(cfg.task).inline_results = False
            task_owner(cfg.task).continue_on_error = True
            task_owner(cfg.task).reuse_prompt_observations = True
            if operation == "generate":
                task_owner(cfg.task).update(self.generation)
        resolve_config_devices(cfg)
        from models.numerics import bind_configuration

        bind_configuration(cfg)
        return cfg

    def cell_path(self, name, key, task):
        return self.report / "cells" / f"{name}--{key}--{task}.json"

    def leaf(self, name, key, task, tag=None, role=None):
        from saebase.runtime.experiment_paths import member_directory

        return member_directory(self.root, name, key, task, tag or self.tag, role)

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
        ranking = "absolute_zscore" if self.ranking == "whiten" else "absolute_native_activation"
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
                and result.get("ranking", "absolute_native_activation") == ranking
                and identity.get("model_sha256") == hashes[0]
                and identity.get("sae_sha256") == hashes[1]
                and len(result.get("top_features", [])) >= self.count
            ):
                found.add(result["prompt_sha256"])
        return all(hashlib.sha256(r["prompt"].encode()).hexdigest() in found for r in rows)

    def source(self, model, alias, cfg, pair, hashes, rows):
        candidates = [
            self.leaf(cfg.dataset.name, model.model, "capture", self.selection_name),
        ]
        if self.capture_source_tag:
            candidates.append(self.leaf(cfg.dataset.name, model.model, "capture", self.capture_source_tag, self.ranking))
        for folder in candidates:
            if self.compatible_source(folder, rows, pair, hashes):
                return folder
        capture_cfg = self.cfg(model, alias, "capture")
        capture_cfg.dataset = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
        capture_cfg.experiment.sub_name = self.selection_name
        from saebase.runtime.experiment_paths import run_leaf

        folder = run_leaf(capture_cfg)
        self.execute(capture_cfg, folder)
        if not self.compatible_source(folder, rows, pair, hashes):
            raise RuntimeError(f"Capture did not produce all requested Top {self.count} rankings: {folder}")
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
        from saebase.runtime.sae_operations import (
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
                from models.numerics import configure_task
                from saebase.runtime.environment import configure_worker

                configure_worker(cfg)
                configure_task(cfg)
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
                    cell = {
                        "dataset": name,
                        "model": key,
                        "task": task,
                        "layer": int(model.layer),
                        "status": "running",
                        "started_at": time.time(),
                    }
                    save(record, cell)
                    try:
                        rows = self.freeze_inputs(cfg, pair)
                        cell["selected_questions"] = len(rows)
                        cell["requested_measurements"] = len(rows) * self.count
                        source = self.source(model, alias, cfg, pair, hashes, rows)
                        task_owner(cfg.task).capture_source = str(source)
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
                    folder = self.leaf(name, model.model, task)
                    cell.pop("run_dir", None)
                    cell.pop("capture_source", None)
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
            "# Intervention and generation of batch progress",
            "",
            f"Sort {self.ranking},Each {self.count} Candidates, value of intervention {self.intervention_value}."
            f"Generate most {self.generation['max_new_tokens']} Character, Temperature {self.generation['temperature']}."
            f"{len(self.plan.models)} A model,{len(self.plan.datasets)} Data sets,"
            f"{len(self.plan.tasks)} A mission..",
            "",
            "Current process:" + str(active),
            "",
            "| Tasks | Qualification experiment completed | Complete Data Set×Model Modules | Current Failed Feature Experiment |",
            "| --- | --- | --- | --- |",
        ]
        for task, total in totals.items():
            lines.append(
                f"| {task} | {total['completed']} | {total['complete_cells']}/120 | {total['failed_measurements']} |"
            )
        lines += [
            "",
            "| Model | Data Set | Tasks | Status | Completed/Number of planned features |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cell in cells:
            if cell["status"] != "pending":
                stat = cell.get("progress", {})
                lines.append(
                    f"| {cell['model']} | {cell['dataset']} | {cell['task']} | {cell['status']} | {stat.get('completed', 0)}/{cell.get('requested_measurements', 'I’ll be ready.')} |"
                )
        (self.report / "progress_zh.md").write_text("\n".join(lines) + "\n")
        return result

    def run(self):
        with (self.report / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env_config = OmegaConf.load(self.root / "conf/base/runtime/runtime.yaml").environments
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
    parser.add_argument("--config", default="conf/tasks/intervene/task.yaml")
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--report")
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
