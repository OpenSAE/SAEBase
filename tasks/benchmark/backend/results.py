"""Atomic, append-only metric ledger with stable keys and a verified manifest."""

from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from saebase.store import fingerprint
from .schema import MetricResult


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".m3-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def result_key(value: MetricResult):
    return fingerprint(
        {
            k: v
            for k, v in value.to_dict().items()
            if k
            not in {
                "status",
                "value",
                "n_valid",
                "reason",
                "warnings",
                "cost",
                "measurement_refs",
                "metadata",
            }
        }
    )


def store_results(folder, results, *, run_id):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() if isinstance(r, MetricResult) else r for r in results]
    for row in rows:
        if row.get("schema") != "saebase.benchmark.metric.v1":
            raise ValueError("invalid metric schema")
    content = "".join(json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in rows).encode()
    digest = hashlib.sha256(content).hexdigest()
    fd, temp = tempfile.mkstemp(prefix=".m3-", dir=root)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, root / "metrics.jsonl")
    finally:
        Path(temp).unlink(missing_ok=True)
    atomic_json(
        root / "metrics_manifest.json",
        {"schema": "saebase.m3.results.v1", "run_id": run_id, "sha256": digest, "rows": len(rows)},
    )
    return root / "metrics.jsonl"


def read_metrics(folder, run_id):
    root = Path(folder)
    manifest = json.loads((root / "metrics_manifest.json").read_text())
    content = (root / "metrics.jsonl").read_bytes()
    if manifest["run_id"] != run_id or hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("metric results integrity or run identity mismatch")
    rows = [json.loads(line) for line in content.splitlines()]
    if len(rows) != manifest["rows"]:
        raise ValueError("metric rows mismatch")
    return rows
