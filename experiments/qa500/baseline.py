"""Run one QA baseline per model/dataset, shared across its two SAE sites."""

from __future__ import annotations

from experiments.qa500.paths import task_dir, typed_file

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.qa500.cohort import PROTOCOL_ID, verify_dataset
from tasks.benchmark.qa_scoring import evaluate_qa
from models.qa_input import make_input


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _existing(path: Path) -> dict[str, dict[str, Any]]:
    previous: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            key = str(item["sample_id"])
            if key in previous or item.get("status") != "executed":
                raise RuntimeError("Duplicate or invalid previous QA baseline row")
            previous[key] = item
    return previous


def run_baseline(
    pair: Any,
    *,
    model_key: str,
    dataset_id: str,
    data_root: Path,
    output_root: Path,
    max_tokens: int = 2048,
    pilot: int | None = None,
) -> dict[str, Any]:
    """Use a verified model with one already loaded real SAE, but no SAE edits."""
    source = verify_dataset(dataset_id, data_root)
    dataset_file = data_root / dataset_id / "questions.jsonl"
    items = [json.loads(line) for line in dataset_file.read_text(encoding="utf-8").splitlines()]
    if pilot is not None:
        if not 1 <= pilot <= len(items):
            raise ValueError("pilot count exceeds frozen cohort")
        items = items[:pilot]
    root = task_dir(output_root, task="benchmark", dataset=dataset_id,
                    model=model_key, initialize=True)
    root.mkdir(parents=True, exist_ok=True)
    name = "pilot" if pilot else "all"
    path = typed_file(root, "metrics", f"{name}.jsonl")
    config = {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": dataset_id,
        "model_key": model_key,
        "input_manifest_sha256": source["sha256"],
        "max_tokens": max_tokens,
        "pilot": pilot,
        "scorer": "qa_inference_v1",
    }
    cfg_file = typed_file(root, "metadata", f"{name}_config.json")
    if cfg_file.exists():
        if json.loads(cfg_file.read_text(encoding="utf-8")) != config:
            raise RuntimeError("QA baseline config changed under an existing run")
    else:
        cfg_file.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    done = _existing(path)
    expected = {item["sample_id"] for item in items}
    if not set(done).issubset(expected):
        raise RuntimeError("Baseline includes records outside selected cohort")
    for item in items:
        if item["sample_id"] in done:
            if done[item["sample_id"]]["prompt_sha256"] != item["prompt_sha256"]:
                raise RuntimeError("Resumed QA baseline prompt identity has changed")
            continue
        try:
            tokens, input_meta = make_input(pair, item, model_key)
            if int(tokens.shape[1]) > max_tokens:
                raise ValueError("token_length_exceeds_limit")
            result = evaluate_qa(pair, tokens, item)
            row = {
                "status": "executed",
                "protocol_id": PROTOCOL_ID,
                "dataset_id": dataset_id,
                "model_key": model_key,
                "sample_id": item["sample_id"],
                "source_revision": source["source_revision"],
                "source_split": source["source_split"],
                **input_meta,
                **result,
            }
            # An answer key is intentionally not echoed into public raw logs.
            _append(path, row)
            done[item["sample_id"]] = row
        except (ValueError, RuntimeError, IndexError) as exc:
            _append(
                typed_file(root, "logs", f"{name}_failures.jsonl"),
                {
                    "status": "failed",
                    "sample_id": item["sample_id"],
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:600],
                },
            )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "model_key": model_key,
        "dataset_id": dataset_id,
        "expected": len(items),
        "executed": len(done),
        "missing": len(items) - len(done),
        "status": "complete" if len(done) == len(items) else "partial",
        "n_scored": sum(row.get("correct") is not None for row in done.values()),
        "n_correct": sum(row.get("correct") is True for row in done.values()),
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        "result_file": str(path),
    }
    denom = summary["n_scored"]
    summary["accuracy"] = summary["n_correct"] / denom if denom else None
    typed_file(root, "reports", f"{name}_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary
