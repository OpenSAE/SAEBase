"""Independent Top N comparisons with persisted progress and validated resume.

Intervene and generate share the same selection and ledger contract. Completed
measurements are reused only when their prompt, feature, resources and protocol
identity match; failures stay explicit and can be retried on resume.
"""

from __future__ import annotations

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
    from tasks.intervene.backend.observations import prompt_observation_session
    from tasks.capture.top_features import capture_top_cohort
    from tasks.sae_runtime import _pair, loaded_pair_session
    from saebase.results import completion_summary
    from saebase.runtime.experiment_results import write_json

    pair = _pair(cfg)
    directory = Path.cwd() / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / f"{operation}_features.jsonl"
    progress = Path.cwd() / "metadata" / f"{operation}_progress.json"
    resuming = cfg.get("runtime", {}).get("mode") == "resume"
    previous = _load_checkpoint(ledger) if resuming and ledger.exists() else {}
    inline = cfg.task.get("inline_results", True)
    samples, states = [], []
    completed = reused = failed = 0
    started = time.monotonic()
    protocol = {
        "operation": operation,
        "value": cfg.task.get("value", 0 if operation == "intervene" else 1),
        "intervention": cfg.task.get("intervention"),
        "seed": int(cfg.seed),
        "max_new_tokens": cfg.task.get("max_new_tokens"),
        "temperature": cfg.task.get("temperature"),
        "verify_identity": cfg.task.get("verify_identity", True),
        "stop_at_context_limit": cfg.task.get("stop_at_context_limit", True)
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
        selected_rows = list(capture_top_cohort(cfg, pair))
        requested = len(selected_rows) * int(cfg.task.get("top_feature_num", 10))
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
                results = []
                with prompt_observation_session(
                    pair, cfg.task.get("reuse_prompt_observations", False)
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
                            child.task.feature_selection = "configured"
                            child.task.feature_id = feature["feature_id"]
                            child.task.prompt = selected["prompt"]
                            if operation == "generate":
                                child.task.conditions = [
                                    {
                                        "name": f"feature_{feature['feature_id']}",
                                        "operation": cfg.task.get("intervention", "add"),
                                        "feature_ids": [feature["feature_id"]],
                                        "value": cfg.task.get("value", 1.0),
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
                                entry["result"] = measure(child, prompt=selected["prompt"])
                                entry["status"] = "completed"
                                completed += 1
                            except Exception as exc:
                                if not cfg.task.get("continue_on_error", False):
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
                            entry["seconds"] = time.monotonic() - began
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
        "feature_ranking": cfg.task.get("feature_ranking", "non_whiten"),
        "top_feature_num": int(cfg.task.get("top_feature_num", 10)),
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
