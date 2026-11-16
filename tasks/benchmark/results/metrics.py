"""Purpose: Construct metric records and save, read and compare their append-only measurement ledger.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from saebase.store import (
    fingerprint,
)
from tasks.benchmark.execution.schema import MetricResult


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


def compare_benchmark_runs(left, right, *, pairs=None) -> dict:
    """Compare verified task ledgers with explicit model/SAE pairing and input proofs."""

    def rows(root):
        values = []
        for path in sorted((Path(root) / "metrics").rglob("metrics_manifest.json")):
            manifest = json.loads(path.read_text())
            values.extend(read_metrics(path.parent, manifest["run_id"]))
        return values

    a, b = rows(left), rows(right)

    def pair(row):
        return (row.get("model_id"), row.get("sae_id"))

    if pairs is None:
        requested_pairs = [
            (identity, identity)
            for identity in sorted({pair(row) for row in a} & {pair(row) for row in b}, key=repr)
        ]
    else:
        requested_pairs = [(tuple(first), tuple(second)) for first, second in pairs]
        if len(set(requested_pairs)) != len(requested_pairs):
            raise ValueError("Duplicate comparison pairs")

    def key(row):
        fields = (
            "metric_id",
            "metric_version",
            "protocol_id",
            "dataset_id",
            "split",
            "seed",
            "sample_unit",
            "condition_id",
            "source_sha256",
            "source_commit",
        )
        return tuple(row.get(field) for field in fields) + (
            json.dumps(row.get("site"), sort_keys=True),
        )

    measurements, used_a, used_b = [], set(), set()
    for first_pair, second_pair in requested_pairs:
        x, y = {}, {}
        for index, row in enumerate(a):
            if pair(row) == first_pair:
                x.setdefault(key(row), []).append((index, row))
        for index, row in enumerate(b):
            if pair(row) == second_pair:
                y.setdefault(key(row), []).append((index, row))
        for identity in sorted(x.keys() & y.keys(), key=repr):
            if len(x[identity]) != 1 or len(y[identity]) != 1:
                continue
            ia, first = x[identity][0]
            ib, second = y[identity][0]
            used_a.add(ia)
            used_b.add(ib)
            proof = ("data_sha256", "protocol_configuration_sha256")
            compatible = all(
                isinstance(first.get("metadata", {}).get(field), str)
                and len(first["metadata"][field]) == 64
                and all(c in "0123456789abcdef" for c in first["metadata"][field])
                and first["metadata"][field] == second.get("metadata", {}).get(field)
                for field in proof
            ) and all(isinstance(value, str) and value for value in (*first_pair, *second_pair))
            valid = compatible and first["status"] == second["status"] == "completed"
            values = first.get("value"), second.get("value")
            delta = (
                values[1] - values[0]
                if valid and all(type(v) in (int, float) for v in values)
                else None
            )
            measurements.append(
                {
                    "left": first,
                    "right": second,
                    "delta": delta,
                    "comparable": bool(compatible),
                    "reason": None
                    if compatible
                    else "Missing or different sample/protocol identity",
                }
            )
    return {
        "supported": True,
        "status": "compared"
        if any(r["delta"] is not None for r in measurements)
        else "not_comparable",
        "measurements": measurements,
        "unmatched_left": [r for i, r in enumerate(a) if i not in used_a],
        "unmatched_right": [r for i, r in enumerate(b) if i not in used_b],
    }


def _metric(
    config,
    target,
    sae_key,
    spec,
    status,
    value=None,
    n_valid=0,
    reason=None,
    condition=None,
    refs=(),
    source_sha=None,
    source_commit=None,
    cost=None,
    metadata=None,
):
    from saebase.store import resolve_positions

    if spec.sample_unit == "token":
        requested = sum(len(resolve_positions(s.record)) for s in config.samples)
    elif spec.sample_unit == "coalition":
        requested = int(config.options.get("sar", {}).get("M", 80))
    elif spec.sample_unit == "variant":
        role = "held_out" if spec.metric_id.endswith("held_out_persistence") else "selection"
        requested = sum(s.record.variant_role == role for s in config.samples)
    elif spec.sample_unit == "dataset":
        requested = 1
    else:
        requested = len(config.samples)
    if n_valid > requested:
        raise ValueError(
            f"metric {spec.metric_id} has {n_valid} valid units > {requested} requested"
        )
    return MetricResult(
        metric_id=spec.metric_id,
        metric_version=spec.version,
        protocol_id=spec.protocol_id,
        model_id=target.model_key,
        sae_id=sae_key,
        site={
            "layer": target.site.layer,
            "hook_name": target.site.hook_name,
            "position": target.site.position,
        },
        dataset_id=config.dataset_id,
        split="mixed"
        if len({s.record.split for s in config.samples}) > 1
        else config.samples[0].record.split,
        seed=config.seed,
        sample_unit=spec.sample_unit,
        n_requested=requested,
        n_valid=n_valid,
        status=status,
        value=value,
        condition_id=condition,
        reason=reason,
        measurement_refs=tuple(refs),
        source_sha256=source_sha,
        source_commit=source_commit,
        cost=cost or {},
        metadata={
            **(metadata or {}),
            "data_sha256": fingerprint(
                [
                    sample.to_dict()
                    if hasattr(sample, "to_dict")
                    else {
                        "record": sample.record.to_dict(),
                        "readout_position": sample.readout_position,
                        "target_token_id": sample.target_token_id,
                        "reference_token_id": sample.reference_token_id,
                    }
                    for sample in config.samples
                ]
            ),
            "protocol_configuration_sha256": fingerprint(
                {
                    "metric": asdict(spec),
                    "conditions": [condition.to_dict() for condition in config.conditions],
                    "options": config.options,
                }
            ),
        },
    )
