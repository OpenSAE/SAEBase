"""QA500-specific baseline and aggregate study measurements."""

from __future__ import annotations

import hashlib
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from models.qa_input import make_input
from scripts.experiments.qa500.cohort import PROTOCOL_ID, verify_dataset
from scripts.experiments.qa500.context import task_dir, typed_file
from tasks.benchmark.qa_metrics import BOOTSTRAP_SEED, summarize_metric
from tasks.benchmark.qa_scoring import evaluate_qa


# --- Shared model baseline ---


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
    root = task_dir(
        output_root, task="benchmark", dataset=dataset_id, model=model_key, initialize=True
    )
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
    typed_file(root, "reports", f"{name}_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    return summary


# --- Frozen SAE benchmark ---


def benchmark_from_capture(
    *,
    data_root: Path,
    result_root: Path,
    model_key: str,
    sae_key: str,
    layer: int,
    dataset_id: str,
) -> dict[str, Any]:
    verify_dataset(dataset_id, data_root)
    capture_dir = task_dir(
        result_root,
        task="capture",
        dataset=dataset_id,
        model=model_key,
        sae_key=sae_key,
        layer=layer,
    )
    base = task_dir(
        result_root,
        task="benchmark",
        dataset=dataset_id,
        model=model_key,
        sae_key=sae_key,
        layer=layer,
        initialize=True,
    )
    capture = typed_file(capture_dir, "metrics", "capture.jsonl")
    if not capture.is_file():
        raise FileNotFoundError("Frozen full capture required before Core benchmark")
    payload = capture.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    sources = {
        obj["sample_id"]: obj
        for obj in (
            json.loads(line)
            for line in (data_root / dataset_id / "questions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    if len(rows) != 500 or len(sources) != 500:
        raise RuntimeError("Core benchmark requires 500 complete frozen questions")
    ids = [row.get("sample_id") for row in rows]
    if len(set(ids)) != 500 or set(ids) != set(sources):
        raise RuntimeError("Core benchmark sample identity mismatch")
    site_id = f"{model_key}:{sae_key}:L{layer}"
    shas = {row.get("sae_sha256") for row in rows}
    if len(shas) != 1 or None in shas:
        raise RuntimeError("Benchmark mixes incompatible SAE checkpoints")
    values = {"core.l0": [], "core.reconstruction_mse_token": []}
    for row in rows:
        source = sources[row["sample_id"]]
        if (
            row.get("status") != "executed"
            or row.get("protocol_id") != PROTOCOL_ID
            or row.get("site_id") != site_id
            or row.get("prompt_sha256") != source["prompt_sha256"]
            or row.get("source_revision") != source["source_revision"]
        ):
            raise RuntimeError("Core benchmark source or site changed")
        l0 = row.get("l0")
        error = row.get("reconstruction_mse")
        if (
            type(l0) is not int
            or l0 < 0
            or l0 > int(row["sae_d_sae"])
            or not isinstance(error, (float, int))
            or not math.isfinite(error)
            or error < 0
        ):
            raise RuntimeError("Core numerical observation invalid")
        values["core.l0"].append(float(l0))
        values["core.reconstruction_mse_token"].append(float(error))
    summary = {
        "status": "complete",
        "execution_status": "executed",
        "protocol_id": PROTOCOL_ID,
        "task": "benchmark",
        "dataset_id": dataset_id,
        "model_key": model_key,
        "sae_key": sae_key,
        "layer": layer,
        "site_id": site_id,
        "sae_sha256": next(iter(shas)),
        "source_capture_sha256": sha256(payload).hexdigest(),
        "sample_count": len(rows),
        "independent_unit": "frozen_question",
        "metrics": [
            {"metric_id": metric, **summarize_metric(vals, seed=BOOTSTRAP_SEED + i)}
            for i, (metric, vals) in enumerate(values.items())
        ],
        "not_claimed": ["saebench_native_fvu", "loss_recovered", "causal_semantics"],
    }
    output = typed_file(base, "reports", "benchmark_summary.json")
    if output.is_file() and json.loads(output.read_text(encoding="utf-8")) != summary:
        raise RuntimeError("Existing benchmark result differs from current frozen observations")
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
