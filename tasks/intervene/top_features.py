"""Independent Top N comparisons with persisted progress and validated resume.

Intervene and generate share the same selection and ledger contract. Completed
measurements are reused only when their prompt, feature, resources and protocol
identity match; failures stay explicit and can be retried on resume.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_owner, task_value

import hashlib
import json
from pathlib import Path
import time

from omegaconf import OmegaConf


def read_feature_results(folder):
    """Read the latest attempt for each measurement from a formal task run."""
    root = Path(folder)
    manifest = json.loads((root / "metadata/manifest.json").read_text())
    operation = manifest["task"]
    rows = {}
    with (root / "reports" / f"{operation}_features.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row.get("measurement_id", (row["sample_id"], row["rank"]))
            rows[key] = row
    return list(rows.values())


def _load_checkpoint(path):
    rows = {}
    data = path.read_bytes()
    offset = 0
    for line in data.splitlines(keepends=True):
        try:
            row = json.loads(line)
            if not row.get("measurement_id"):
                raise ValueError("Feature checkpoint has no measurement identity")
        except (ValueError, KeyError):
            if offset + len(line) == len(data) and not line.endswith(b"\n"):
                # A killed writer may leave a partial final line, never a middle line.
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise
        rows[row["measurement_id"]] = row
        offset += len(line)
        if not line.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def run_top_features(cfg, operation, measure):
    from tasks.intervene.editing.observations import prompt_observation_session
    from tasks.capture.top_features import capture_top_cohort
    from saebase.runtime.sae_operations import _pair, loaded_pair_session
    from saebase.results import completion_summary
    from saebase.runtime.experiment_results import write_json
    from saebase.runtime.performance import settings as performance_settings, current, stage

    pair = _pair(cfg)
    directory = Path.cwd() / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / f"{operation}_features.jsonl"
    progress = Path.cwd() / "metadata" / f"{operation}_progress.json"
    resuming = cfg.get("runtime", {}).get("mode") == "resume"
    previous = _load_checkpoint(ledger) if resuming and ledger.exists() else {}
    inline = task_value(cfg.task, "inline_results", True)
    samples, states = [], []
    completed = reused = failed = 0
    started = time.monotonic()
    protocol = {
        "operation": operation,
        "value": task_value(cfg.task, "value", 0 if operation == "intervene" else 1),
        "intervention": task_value(cfg.task, "intervention"),
        "seed": int(cfg.seed),
        "max_new_tokens": task_value(cfg.task, "max_new_tokens"),
        "temperature": task_value(cfg.task, "temperature"),
        "verify_identity": task_value(cfg.task, "verify_identity", True),
        "stop_at_context_limit": task_value(cfg.task, "stop_at_context_limit", True)
        if operation == "generate"
        else None,
    }

    def measurement_id(selected, rank, feature):
        identity = {
            "protocol": protocol,
            "sample_id": selected["sample_id"],
            "prompt_sha256": selected["prompt_sha256"],
            "rank": rank,
            "feature_id": feature["feature_id"],
            "ranking": selected["ranking"],
            "normalization": selected["normalization"],
            "input_identity": selected["input_identity"],
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    with loaded_pair_session(pair):
        with stage("feature_selection"):
            selected_rows = list(capture_top_cohort(cfg, pair))
        requested = len(selected_rows) * int(task_value(cfg.task, "top_feature_num", 10))
        valid_ids = {
            measurement_id(row, rank, feature)
            for row in selected_rows
            for rank, feature in enumerate(row["top_features"], 1)
        }
        if set(previous) - valid_ids:
            raise ValueError(
                "Feature checkpoint contains measurements outside this cohort/protocol"
            )
        with ledger.open("a" if resuming else "x") as stream:
            for selected in selected_rows:
                sample_started = time.monotonic()
                prepared = None
                prepared_seconds = None
                if performance_settings(cfg)["enabled"] and not resuming:
                    from tasks.intervene.performance import measure_features

                    try:
                        prepared, _ = measure_features(cfg, operation, selected, pair, measure)
                        prepared_seconds = (time.monotonic() - sample_started) / len(
                            selected["top_features"]
                        )
                    except Exception as exc:
                        from saebase.api import UnsupportedCapability

                        if not isinstance(exc, UnsupportedCapability) and not task_value(cfg.task, "continue_on_error", False
                        ):
                            raise
                        if current() is not None:
                            current().experiments.append(
                                {
                                    "optimization_fallback": f"{type(exc).__name__}: {exc}",
                                    "execution": "original per-feature loop",
                                }
                            )
                results = []
                with prompt_observation_session(
                    pair, task_value(cfg.task, "reuse_prompt_observations", False)
                ):
                    for rank, feature in enumerate(selected["top_features"], 1):
                        uid = measurement_id(selected, rank, feature)
                        entry = previous.get(uid)
                        if entry and entry.get("status", "completed") == "completed":
                            reused += 1
                            completed += 1
                            state = {"status": "completed"}
                        else:
                            child = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                            task_owner(child.task).feature_selection = "configured"
                            task_owner(child.task).feature_id = feature["feature_id"]
                            child.task.prompt = selected["prompt"]
                            if operation == "generate":
                                task_owner(child.task).conditions = [
                                    {
                                        "name": f"feature_{feature['feature_id']}",
                                        "operation": task_value(cfg.task, "intervention", "add"),
                                        "feature_ids": [feature["feature_id"]],
                                        "value": task_value(cfg.task, "value", 1.0),
                                    }
                                ]
                            entry = {
                                "measurement_id": uid,
                                "sample_id": selected["sample_id"],
                                "prompt": selected["prompt"],
                                "prompt_sha256": selected["prompt_sha256"],
                                "rank": rank,
                                "feature": feature,
                                "selection": {
                                    k: selected[k]
                                    for k in (
                                        "ranking",
                                        "normalization",
                                        "input_identity",
                                        "capture_source",
                                    )
                                },
                            }
                            began = time.monotonic()
                            try:
                                entry["result"] = (
                                    prepared[rank - 1]
                                    if prepared is not None
                                    else measure(child, prompt=selected["prompt"])
                                )
                                entry["status"] = "completed"
                                completed += 1
                            except Exception as exc:
                                if not task_value(cfg.task, "continue_on_error", False):
                                    raise
                                import gc
                                import torch

                                entry.update(
                                    status="failed",
                                    error={"type": type(exc).__name__, "message": str(exc)},
                                )
                                failed += 1
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            entry["seconds"] = (
                                prepared_seconds
                                if prepared_seconds is not None
                                else time.monotonic() - began
                            )
                            if prepared_seconds is not None:
                                entry["seconds_kind"] = (
                                    "amortized question compute, including any tuning, divided by feature count"
                                )
                            stream.write(
                                json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            stream.flush()
                            state = {"status": entry["status"]}
                            if entry["status"] == "failed":
                                state["reason"] = entry["error"]["message"]
                        states.append(state)
                        if inline:
                            results.append(entry)
                        write_json(
                            progress,
                            {
                                "operation": operation,
                                "requested": requested,
                                "completed": completed,
                                "reused": reused,
                                "failed": failed,
                                "sample_id": selected["sample_id"],
                                "feature_rank": rank,
                                "elapsed_seconds": time.monotonic() - started,
                                "samples_path": str(ledger),
                            },
                        )
                samples.append(
                    {
                        "sample_id": selected["sample_id"],
                        "prompt": selected["prompt"],
                        **(
                            {"features": results}
                            if inline
                            else {"feature_count": len(selected["top_features"])}
                        ),
                    }
                )
                if current() is not None:
                    current().sample(
                        selected["sample_id"],
                        time.monotonic() - sample_started,
                        feature_comparisons=len(selected["top_features"]),
                        timing_scope="whole question including tuning and ledger I/O; feature selection timed separately",
                    )
        if set(previous) - valid_ids:
            raise ValueError(
                "Feature checkpoint contains measurements outside this cohort/protocol"
            )
    artifacts = [ledger]
    artifacts.extend((Path.cwd() / "metadata").glob("capture_normalization.*"))
    return {
        "status": "executed",
        "operation": operation,
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "layer": pair.site.layer,
        "site": pair.site.hook_name,
        "feature_selection": "capture_top",
        "feature_ranking": task_value(cfg.task, "feature_ranking", "non_whiten"),
        "top_feature_num": int(task_value(cfg.task, "top_feature_num", 10)),
        "independent_feature_edits": True,
        "samples": samples,
        "samples_path": str(ledger),
        "reused_measurements": reused,
        "input_scope": {
            "unit": "prompt",
            "requested": len(selected_rows),
            "processed": len(samples),
        },
        "completion": completion_summary(states, expected=requested, unit="feature_comparison"),
        "required_artifacts": [
            {
                "path": str(p.relative_to(Path.cwd())),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in artifacts
        ],
    }
