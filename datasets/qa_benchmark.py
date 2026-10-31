"""Independent standard QA500 Core benchmarking from actual captured records.

This is a small, explicitly named Core protocol, not a substitute for the
native SAEBench estimator or LM loss-recovered. Bootstrap samples the
independent frozen question IDs, never tokens or duplicate variants.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any

from saebase.datasets.qa_standard import PROTOCOL_ID, verify_dataset

BOOTSTRAP_SEED = 20261031
BOOTSTRAP_DRAWS = 512


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left, right = int(position), min(int(position) + 1, len(ordered) - 1)
    return float(ordered[left] + (ordered[right] - ordered[left]) * (position - left))


def summarize_metric(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Core benchmark requires finite nonempty measurements")
    n = len(values)
    rnd = random.Random(seed)
    bootstrap = [
        sum(values[rnd.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "value": float(sum(values) / n),
        "n_valid": n,
        "ci_lower_95": _percentile(bootstrap, 0.025),
        "ci_upper_95": _percentile(bootstrap, 0.975),
        "bootstrap_seed": seed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "confidence_method": "iid_question_percentile_bootstrap_v1",
    }


def benchmark_from_capture(
    *, data_root: Path, result_root: Path,
    model_key: str, sae_key: str, layer: int, dataset_id: str,
) -> dict[str, Any]:
    verify_dataset(dataset_id, data_root)
    base = result_root / PROTOCOL_ID / model_key / sae_key / f"L{layer}" / dataset_id
    capture = base / "capture.jsonl"
    if not capture.is_file():
        raise FileNotFoundError("Frozen full capture required before Core benchmark")
    payload = capture.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    sources = {
        obj["sample_id"]: obj
        for obj in (
            json.loads(line) for line in
            (data_root / dataset_id / "questions.jsonl").read_text(encoding="utf-8").splitlines()
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
            type(l0) is not int or l0 < 0 or l0 > int(row["sae_d_sae"])
            or not isinstance(error, (float, int)) or not math.isfinite(error)
            or error < 0
        ):
            raise RuntimeError("Core numerical observation invalid")
        values["core.l0"].append(float(l0))
        values["core.reconstruction_mse_token"].append(float(error))
    summary = {
        "status": "executed", "protocol_id": PROTOCOL_ID,
        "task": "benchmark", "dataset_id": dataset_id,
        "model_key": model_key, "sae_key": sae_key,
        "layer": layer, "site_id": site_id,
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
    output = base / "benchmark_summary.json"
    if output.is_file() and json.loads(output.read_text(encoding="utf-8")) != summary:
        raise RuntimeError("Existing benchmark result differs from current frozen observations")
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
