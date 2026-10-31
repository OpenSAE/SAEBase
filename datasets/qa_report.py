"""Independent QA500 result auditor and Hugging Face staging exporter.

The exporter never writes to the existing Evaluation repository. It writes
new, clearly labeled subset CSVs only after auditing exact frozen identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from saebase.datasets.qa_standard import DATASETS, PROTOCOL_ID, N_PER_DATASET, verify_dataset
from saebase.datasets.qa_pipeline import site_list
from saebase.datasets.qa_batch import TASK_STATUS


DEFAULT_INPUT = Path("experiment/qa500/inputs")
DEFAULT_RESULTS = Path("experiment/qa500/results")
DEFAULT_EXPORT = Path("experiment/qa500/hf_stage")
CONFIGS = (
    "qa_v1_inputs",
    "qa_v1_sites",
    "qa_v1_runs",
    "qa_v1_metrics",
    "qa_v1_per_question",
    "qa_v1_task_records",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ids_exact(rows: list[dict[str, Any]], expected: set[str]) -> bool:
    ids = [row.get("sample_id") for row in rows]
    return len(ids) == len(expected) == len(set(ids)) and set(ids) == expected


def _check_capture(row: dict[str, Any], *, source: dict[str, Any], site: dict[str, Any]) -> None:
    if (
        row.get("status") != "executed"
        or row.get("protocol_id") != PROTOCOL_ID
        or row.get("sample_id") != source["sample_id"]
        or row.get("prompt_sha256") != source["prompt_sha256"]
        or row.get("source_revision") != source["source_revision"]
        or row.get("site_id") != site["site_id"]
        or row.get("hook_name") != site["hook_name"]
        or row.get("sae_key") != site["sae_key"]
    ):
        raise RuntimeError("Capture provenance mismatch")
    d_in = int(row.get("sae_d_in", 0))
    d_sae = int(row.get("sae_d_sae", 0))
    k = int(row.get("l0", -1))
    if d_in < 1 or d_sae < 1 or k < 0 or k > d_sae:
        raise RuntimeError("Invalid actual SAE width or active feature count")
    if row.get("sae_sha256") is None or len(row["sae_sha256"]) != 64:
        raise RuntimeError("Missing SAE weight digest")
    if row.get("token_count", 0) < 2 or row.get("token_count", 0) > 2048:
        raise RuntimeError("Token length violates fixed 2048-token protocol")
    if row.get("capture_position") != -1 or not row.get("baseline_logits_finite"):
        raise RuntimeError("Capture readout position or baseline logits invalid")
    for measure in ("reconstruction_mse", "activation_norm_l2"):
        val = row.get(measure)
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
            raise RuntimeError(f"Invalid numerical measurement {measure}")
    features = row.get("top_features")
    if not isinstance(features, list) or len(features) > 10 or len(features) > k:
        raise RuntimeError("Top feature count inconsistent with L0")
    ids = []
    abs_values = []
    for f in features:
        fid = f.get("feature_id")
        activation = f.get("activation")
        if (
            type(fid) is not int
            or not 0 <= fid < d_sae
            or not isinstance(activation, (int, float))
            or not math.isfinite(activation)
            or activation == 0
        ):
            raise RuntimeError("Invalid active feature ID or activation")
        if f.get("explanation_status") not in {
            "unmapped",
            "checkpoint_mismatch",
            "not_available",
            "no_description",
            "source_reported",
        }:
            raise RuntimeError("Unknown explanation provenance status")
        if f["explanation_status"] != "source_reported" and f.get("explanation") is not None:
            raise RuntimeError("Unverified Neuronpedia explanation")
        ids.append(fid)
        abs_values.append(abs(activation))
    if len(set(ids)) != len(ids) or abs_values != sorted(abs_values, reverse=True):
        raise RuntimeError("Top features must be unique, ordered by absolute activation")


def audit(data_root: Path, result_root: Path) -> dict[str, Any]:
    """Independent checks; incomplete records remain recorded as incomplete."""
    sites = site_list()
    dataset_sources = {}
    defects: list[str] = []
    counts = {
        "input_datasets_complete": 0,
        "baseline_complete": 0,
        "capture_complete": 0,
        "benchmark_complete": 0,
        "intervene_complete": 0,
        "generate_complete": 0,
    }
    for dataset in DATASETS:
        try:
            verify_dataset(dataset, data_root)
            source_rows = _read_rows(data_root / dataset / "questions.jsonl")
            dataset_sources[dataset] = {r["sample_id"]: r for r in source_rows}
            if len(dataset_sources[dataset]) != N_PER_DATASET:
                raise ValueError("Frozen input IDs not unique")
            counts["input_datasets_complete"] += 1
        except Exception as exc:
            defects.append(f"inputs/{dataset}: {exc}")
    for model_key in dict.fromkeys(s["model_key"] for s in sites):
        for dataset in DATASETS:
            sources = dataset_sources.get(dataset)
            if sources is None:
                continue
            path = (
                result_root / PROTOCOL_ID / "_model_baselines" / model_key / dataset / "all.jsonl"
            )
            rows = _read_rows(path)
            if not _ids_exact(rows, set(sources)):
                defects.append(f"baseline/{model_key}/{dataset}: incomplete or duplicate IDs")
                continue
            if any(
                r.get("status") != "executed"
                or r.get("prompt_sha256") != sources[r["sample_id"]]["prompt_sha256"]
                or r.get("model_key") != model_key
                or r.get("protocol_id") != PROTOCOL_ID
                for r in rows
            ):
                defects.append(f"baseline/{model_key}/{dataset}: provenance mismatch")
            else:
                counts["baseline_complete"] += 1
    for site in sites:
        for dataset in DATASETS:
            sources = dataset_sources.get(dataset)
            if sources is None:
                continue
            base = (
                result_root
                / PROTOCOL_ID
                / site["model_key"]
                / site["sae_key"]
                / f"L{site['layer']}"
                / dataset
            )
            rows = _read_rows(base / "capture.jsonl")
            if not _ids_exact(rows, set(sources)):
                defects.append(f"{site['site_id']}/{dataset}: incomplete capture IDs")
                continue
            try:
                shas = set()
                for r in rows:
                    _check_capture(r, source=sources[r["sample_id"]], site=site)
                    shas.add(r["sae_sha256"])
                if len(shas) != 1:
                    raise RuntimeError("Multiple SAE checkpoint weights within one site/dataset")
                counts["capture_complete"] += 1
                benchmark_file = base / "benchmark_summary.json"
                if not benchmark_file.is_file():
                    defects.append(f"{site['site_id']}/{dataset}: benchmark summary missing")
                else:
                    summary = json.loads(benchmark_file.read_text(encoding="utf-8"))
                    from hashlib import sha256

                    metric_ids = {
                        m.get("metric_id") for m in summary.get("metrics", [])
                    }
                    if (
                        summary.get("status") != "complete"
                        or summary.get("source_capture_sha256")
                        != sha256((base / "capture.jsonl").read_bytes()).hexdigest()
                        or summary.get("site_id") != site["site_id"]
                        or summary.get("sample_count") != 500
                        or metric_ids != {"core.l0", "core.reconstruction_mse_token"}
                    ):
                        defects.append(f"{site['site_id']}/{dataset}: benchmark identity mismatch")
                    else:
                        counts["benchmark_complete"] += 1
            except (RuntimeError, ValueError, KeyError) as exc:
                defects.append(f"{site['site_id']}/{dataset}: {exc}")
            for task, n in (("intervene", 20), ("generate", 5)):
                effects = _read_rows(base / f"{task}.jsonl")
                if len(effects) != n or len({r.get("sample_id") for r in effects}) != n:
                    defects.append(f"{site['site_id']}/{dataset}/{task}: {len(effects)} of {n}")
                    continue
                if any(
                    r.get("status") != "executed"
                    or r.get("protocol_id") != PROTOCOL_ID
                    or r.get("sae_key") != site["sae_key"]
                    or r.get("layer") != site["layer"]
                    or r.get("feature_selection") != "max_abs_nonzero_activation_on_same_input_v1"
                    for r in effects
                ):
                    defects.append(f"{site['site_id']}/{dataset}/{task}: invalid effects")
                else:
                    counts[f"{task}_complete"] += 1
    expected = {
        "input_datasets_complete": 10,
        "baseline_complete": 100,
        "capture_complete": 200,
        "benchmark_complete": 200,
        "intervene_complete": 200,
        "generate_complete": 200,
    }
    status = "complete" if counts == expected and not defects else "partial"
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "counts": counts,
        "expected": expected,
        "defect_count": len(defects),
        "defects": defects,
        "review_boundary": {
            "explain": "annotation-only, no native explanation validation",
            "circuits": "not part of the universal 20-site matrix",
            "train": "requires independent training corpus and native approval",
            "math": "MATH500 equivalence scorer pending unless separate verified grader",
        },
    }


def _csv(path: Path, fields: list[str], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    name: json.dumps(rec[name], ensure_ascii=False, sort_keys=True)
                    if isinstance(rec.get(name), (dict, list))
                    else rec.get(name)
                    for name in fields
                }
            )


def stage(
    data_root: Path, result_root: Path, output: Path, *, require_complete: bool = True
) -> dict[str, Any]:
    """Prepare new, source-safe tables in a separate local staging directory."""
    report = audit(data_root, result_root)
    if require_complete and report["status"] != "complete":
        raise RuntimeError("Cannot stage complete HF release from partial QA500 records")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    inputs = []
    for dataset in DATASETS:
        for row in _read_rows(data_root / dataset / "questions.jsonl"):
            inputs.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "dataset_id": dataset,
                    "sample_id": row["sample_id"],
                    "source_revision": row["source_revision"],
                    "source_repo": row["source_repo"],
                    "source_split": row["source_split"],
                    "prompt_sha256": row["prompt_sha256"],
                    "restricted_source": row["restricted_source"],
                }
            )
    _csv(output / "qa_v1_inputs/data.csv", list(inputs[0]), inputs)
    sites = site_list()
    _csv(output / "qa_v1_sites/data.csv", list(sites[0]), sites)
    model_baselines = []
    per_question = []
    metrics = []
    task_records = []
    for model in dict.fromkeys(s["model_key"] for s in sites):
        for dataset in DATASETS:
            p = result_root / PROTOCOL_ID / "_model_baselines" / model / dataset / "all.jsonl"
            data = _read_rows(p)
            if not data:
                continue
            complete = sum(x.get("correct") is not None for x in data)
            metrics.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "model_key": model,
                    "sae_key": "",
                    "layer": "",
                    "dataset_id": dataset,
                    "task": "qa_baseline",
                    "metric_id": "qa.accuracy",
                    "value": sum(x.get("correct") is True for x in data) / complete
                    if complete
                    else "",
                    "n_valid": complete,
                    "n_expected": N_PER_DATASET,
                    "status": "complete" if len(data) == N_PER_DATASET else "partial",
                }
            )
            model_baselines.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "model_key": model,
                    "dataset_id": dataset,
                    "n": len(data),
                    "file_sha256": _sha(p),
                }
            )
    for site in sites:
        for dataset in DATASETS:
            base = (
                result_root
                / PROTOCOL_ID
                / site["model_key"]
                / site["sae_key"]
                / f"L{site['layer']}"
                / dataset
            )
            p = base / "capture.jsonl"
            data = _read_rows(p)
            if data:
                metrics.extend(
                    [
                        {
                            "protocol_id": PROTOCOL_ID,
                            "model_key": site["model_key"],
                            "sae_key": site["sae_key"],
                            "layer": site["layer"],
                            "dataset_id": dataset,
                            "task": "capture",
                            "metric_id": metric,
                            "value": (sum(r[field] for r in data) / len(data)),
                            "n_valid": len(data),
                            "n_expected": N_PER_DATASET,
                            "status": "complete" if len(data) == N_PER_DATASET else "partial",
                        }
                        for metric, field in (
                            ("core.l0", "l0"),
                            ("core.reconstruction_mse_token", "reconstruction_mse"),
                        )
                    ]
                )
                for r in data:
                    per_question.append(
                        {
                            "protocol_id": PROTOCOL_ID,
                            "sample_id": r["sample_id"],
                            "model_key": r["model_key"],
                            "dataset_id": dataset,
                            "sae_key": r["sae_key"],
                            "layer": r["layer"],
                            "site_id": r["site_id"],
                            "sae_sha256": r["sae_sha256"],
                            "prompt_sha256": r["prompt_sha256"],
                            "token_ids_sha256": r.get("token_ids_sha256", ""),
                            "l0": r["l0"],
                            "reconstruction_mse": r["reconstruction_mse"],
                            "top_features": r["top_features"],
                            "status": r["status"],
                        }
                    )
            benchmark_file = base / "benchmark_summary.json"
            if benchmark_file.is_file():
                benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
                for m in benchmark["metrics"]:
                    metrics.append({
                        "protocol_id": PROTOCOL_ID,
                        "model_key": site["model_key"],
                        "sae_key": site["sae_key"],
                        "layer": site["layer"],
                        "dataset_id": dataset,
                        "task": "benchmark",
                        "metric_id": m["metric_id"],
                        "value": m["value"],
                        "n_valid": m["n_valid"],
                        "n_expected": N_PER_DATASET,
                        "ci_lower_95": m["ci_lower_95"],
                        "ci_upper_95": m["ci_upper_95"],
                        "status": benchmark["status"],
                    })
            for task in ("intervene", "generate"):
                efile = base / f"{task}.jsonl"
                for r in _read_rows(efile):
                    task_records.append(
                        {
                            "protocol_id": PROTOCOL_ID,
                            "task": task,
                            "model_key": site["model_key"],
                            "sae_key": site["sae_key"],
                            "layer": site["layer"],
                            "dataset_id": dataset,
                            "sample_id": r["sample_id"],
                            "feature_id": r["feature_id"],
                            "result": r["measurement"],
                            "status": r["status"],
                        }
                    )
    # Explanation labels are source-reported proposals, with deterministic
    # discovery and held-out top-activation occurrence, not semantic validation.
    for site in sites:
        for dataset in DATASETS:
            ep = (
                result_root
                / PROTOCOL_ID
                / site["model_key"]
                / site["sae_key"]
                / f"L{site['layer']}"
                / dataset
                / "explain.json"
            )
            if not ep.is_file():
                continue
            obj = json.loads(ep.read_text(encoding="utf-8"))
            if (
                obj.get("status") != "source_annotations_only"
                or obj.get("site_id") != site["site_id"]
            ):
                raise RuntimeError("Unrecognized explanation artifact")
            for entry in obj["candidates"]:
                task_records.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "task": "explain",
                        "model_key": site["model_key"],
                        "sae_key": site["sae_key"],
                        "layer": site["layer"],
                        "dataset_id": dataset,
                        "sample_id": "",
                        "feature_id": entry["feature_id"],
                        "result": entry,
                        "status": "source_annotations_only",
                    }
                )
    native_root = result_root / PROTOCOL_ID / "_native" / "circuits_gemma2_2b"
    for receipt in sorted(native_root.glob("*.json")):
        if receipt.name == "summary.json":
            continue
        obj = json.loads(receipt.read_text(encoding="utf-8"))
        task_records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "task": "circuits",
                "model_key": "gemma2_2b_base",
                "sae_key": "",
                "layer": "",
                "dataset_id": "boolq",
                "sample_id": obj.get("sample_id", ""),
                "feature_id": "",
                "result": obj,
                "status": obj.get("status", "blocked"),
            }
        )
    native_train = result_root.parent / "native_train" / "summary.json"
    if native_train.is_file():
        obj = json.loads(native_train.read_text(encoding="utf-8"))
        task_records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "task": "train",
                "model_key": "gemma2_2b_base",
                "sae_key": "",
                "layer": 13,
                "dataset_id": "wikitext",
                "sample_id": "",
                "feature_id": "",
                "result": obj,
                "status": obj.get("status", "blocked"),
            }
        )
    # Preserve explicit coverage gaps rather than creating fake zero metrics.
    for task in ("explain", "circuits", "train"):
        if any(item["task"] == task for item in task_records):
            continue
        task_records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "task": task,
                "model_key": "",
                "sae_key": "",
                "layer": "",
                "dataset_id": "",
                "sample_id": "",
                "feature_id": "",
                "result": {"reason": TASK_STATUS[task]},
                "status": "not_verified",
            }
        )
    run_records = model_baselines + [
        {
            "protocol_id": PROTOCOL_ID,
            "model_key": s["model_key"],
            "dataset_id": d,
            "n": len(
                _read_rows(
                    result_root
                    / PROTOCOL_ID
                    / s["model_key"]
                    / s["sae_key"]
                    / f"L{s['layer']}"
                    / d
                    / "capture.jsonl"
                )
            ),
            "file_sha256": (
                _sha(
                    result_root
                    / PROTOCOL_ID
                    / s["model_key"]
                    / s["sae_key"]
                    / f"L{s['layer']}"
                    / d
                    / "capture.jsonl"
                )
                if (
                    result_root
                    / PROTOCOL_ID
                    / s["model_key"]
                    / s["sae_key"]
                    / f"L{s['layer']}"
                    / d
                    / "capture.jsonl"
                ).is_file()
                else ""
            ),
        }
        for s in sites
        for d in DATASETS
    ]
    _csv(
        output / "qa_v1_runs/data.csv",
        ["protocol_id", "model_key", "dataset_id", "n", "file_sha256"],
        run_records,
    )
    _csv(
        output / "qa_v1_metrics/data.csv",
        [
            "protocol_id",
            "model_key",
            "sae_key",
            "layer",
            "dataset_id",
            "task",
            "metric_id",
            "value",
            "n_valid",
            "n_expected",
            "status",
        ],
        metrics,
    )
    _csv(
        output / "qa_v1_per_question/data.csv",
        [
            "protocol_id",
            "sample_id",
            "model_key",
            "dataset_id",
            "sae_key",
            "layer",
            "site_id",
            "sae_sha256",
            "prompt_sha256",
            "token_ids_sha256",
            "l0",
            "reconstruction_mse",
            "top_features",
            "status",
        ],
        per_question,
    )
    _csv(
        output / "qa_v1_task_records/data.csv",
        [
            "protocol_id",
            "task",
            "model_key",
            "sae_key",
            "layer",
            "dataset_id",
            "sample_id",
            "feature_id",
            "result",
            "status",
        ],
        task_records,
    )
    evidence = {
        "protocol_id": PROTOCOL_ID,
        "status": report["status"],
        "publication_approved": False,
        "source_text_included": False,
        "number_of_subsets": len(CONFIGS),
        "file_sha256": {name: _sha(output / name / "data.csv") for name in CONFIGS},
    }
    (output / "publication_manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent QA500 audit and HF staging")
    parser.add_argument("action", choices=["audit", "stage"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "audit":
        result = audit(args.data_root, args.result_root)
    else:
        result = stage(
            args.data_root,
            args.result_root,
            args.output,
            require_complete=not args.allow_partial,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(args.action == "audit" and result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
