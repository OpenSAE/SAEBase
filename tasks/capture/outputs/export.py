"""Purpose and main functions: Export saved capture observations, shared explanations and normalization artifacts to Parquet.

Scope: Explicitly selected native and whiten final-token capture matrices. Reads existing
Task records and writes a named dataset directory for local use or publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tasks.capture.annotations import read_samples


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def schemas():
    import pyarrow as pa

    text, integer, number, boolean = pa.string(), pa.int64(), pa.float64(), pa.bool_()

    def schema(fields):
        return pa.schema(fields, metadata={b"protocol": b"saebase.capture.parquet.v1"})

    common = [("run_id", text), ("model_key", text), ("dataset", text), ("mode", text)]
    explanation = pa.struct(
        [
            (key, text)
            for key in (
                "id",
                "description",
                "explainer_model",
                "method",
                "author_id",
                "details_json",
            )
        ]
    )
    return {
        "runs": schema(
            common
            + [
                (key, text)
                for key in (
                    "sae_key",
                    "hook_name",
                    "site_uid",
                    "model_sha256",
                    "sae_sha256",
                    "source_commit",
                    "source_worktree_sha256",
                    "source_run",
                    "source_manifest_sha256",
                    "samples_sha256",
                    "annotations_sha256",
                    "normalization_uid",
                    "source_status",
                    "capture_status",
                    "annotation_status",
                )
            ]
            + [
                (key, integer)
                for key in (
                    "layer",
                    "sample_count",
                    "observation_count",
                    "top_explained_sample_count",
                )
            ]
        ),
        "samples": schema(
            common
            + [
                (key, text)
                for key in (
                    "sample_uid",
                    "sample_id",
                    "source_split",
                    "source_index",
                    "prompt_sha256",
                    "source_status",
                )
            ]
            + [
                (key, integer)
                for key in ("token_position", "active_count", "top_count", "top_explained_count")
            ]
            + [("top_feature_uids", pa.list_(text)), ("top_features_explained", boolean)]
        ),
        "activations": schema(
            common
            + [
                ("sample_uid", text),
                ("feature_uid", text),
                ("feature_id", integer),
                ("token_position", integer),
            ]
            + [(key, number) for key in ("activation", "zscore", "mean", "std")]
            + [
                (key, boolean)
                for key in ("is_top", "is_active", "zero_native_activation", "whiten_eligible")
            ]
            + [("top_rank", integer), ("active_rank", integer)]
        ),
        "features": schema(
            [
                (key, text)
                for key in (
                    "feature_uid",
                    "site_uid",
                    "model_key",
                    "sae_key",
                    "model_sha256",
                    "sae_sha256",
                    "hook_name",
                    "neuronpedia_url",
                    "annotation_status",
                )
            ]
            + [
                ("layer", integer),
                ("feature_id", integer),
                ("explanation_count", integer),
                ("explanations", pa.list_(explanation)),
                ("annotation_exports", pa.list_(text)),
                ("observed_statuses", pa.list_(text)),
            ]
        ),
        "normalization": schema(
            common
            + [
                (key, text)
                for key in (
                    "normalization_uid",
                    "protocol_id",
                    "cohort_sha256",
                    "array_file",
                    "metadata_file",
                )
            ]
            + [
                (key, integer)
                for key in ("token_count", "prompt_count", "feature_count", "valid_feature_count")
            ]
        ),
    }


class Rows:
    def __init__(self, path, schema):
        import pyarrow.parquet as pq

        self.path, self.schema = path, schema
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.buffer = []
        self.count = 0

    def append(self, row):
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= 25000:
            self.flush()

    def flush(self):
        import pyarrow as pa

        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self):
        self.flush()
        self.writer.close()


def write_rows(path, rows, schema):
    writer = Rows(path, schema)
    try:
        for row in rows:
            writer.append(row)
    finally:
        writer.close()


def _mode(result):
    identity = result.get("input_identity") or {}
    declared = result.get("mode") or identity.get("execution", {}).get("capture_mode")
    if declared:
        return declared
    protocol = identity.get("protocol_id", "")
    if protocol in {"saebase.capture.v1", "saebase.capture.dataset.v1"}:
        return "non_whiten"
    raise ValueError("Capture mode is absent from the saved input identity")


def _explanation(value):
    result = {
        key: str(value[key]) if value.get(key) is not None else None
        for key in ("id", "description", "explainer_model", "method", "author_id")
    }
    result["details_json"] = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return result


def _add_feature(features, site, feature, annotation_export):
    index = int(feature["feature_id"])
    uid = site["site_uid"] + ":" + str(index)
    entry = features.setdefault(
        uid,
        {
            **site,
            "feature_uid": uid,
            "feature_id": index,
            "neuronpedia_url": None,
            "_explanations": {},
            "_exports": set(),
            "_statuses": set(),
        },
    )
    annotation = feature.get("neuronpedia") or {}
    entry["neuronpedia_url"] = annotation.get("url") or entry["neuronpedia_url"]
    entry["_statuses"].add(annotation.get("status", "unexplained"))
    if annotation_export:
        entry["_exports"].add(annotation_export)
    for description in annotation.get("explanations", []):
        entry["_explanations"][digest(description)] = _explanation(description)
    return uid


def _run(root, record, expected_mode, output, types, features, project_root):
    manifest_path = root / "metadata/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = (manifest.get("result") or {}).get("summary") or {}
    ledger = root / "reports/capture_samples.jsonl"
    ledger_sha = file_digest(ledger) if ledger.exists() else None
    run_id = digest(
        {
            "mode": expected_mode,
            "dataset": record["dataset"],
            "model": record["model"],
            "run": root.name,
            "samples": ledger_sha,
        }
    )
    common = {
        "run_id": run_id,
        "model_key": record["model"],
        "dataset": record["dataset"],
        "mode": expected_mode,
    }
    source = manifest.get("source") or {}
    supplement = root / "reports/capture_explanations.json"
    report = {
        **common,
        "sae_key": record.get("sae") or summary.get("sae_key"),
        "layer": record.get("layer", summary.get("layer")),
        "hook_name": None,
        "site_uid": None,
        "source_commit": source.get("commit"),
        "source_worktree_sha256": digest(source.get("working_tree")),
        "source_run": root.relative_to(project_root).as_posix(),
        "source_manifest_sha256": file_digest(manifest_path),
        "samples_sha256": ledger_sha,
        "annotations_sha256": file_digest(supplement) if supplement.exists() else None,
        "normalization_uid": None,
        "source_status": manifest.get("status"),
        "capture_status": "completed" if ledger.exists() else manifest.get("status"),
        "annotation_status": None,
        "sample_count": 0,
        "observation_count": 0,
        "top_explained_sample_count": 0,
    }
    normalization = None
    norm = root / "metadata/capture_normalization.json"
    if norm.exists():
        metadata = json.loads(norm.read_text())
        identity = metadata["identity"]
        uid = identity["content_sha256"]
        report["normalization_uid"] = uid
        folder = output / "normalization" / run_id
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("capture_normalization.npz", "capture_normalization.json"):
            shutil.copy2(root / "metadata" / name, folder / name)
        normalization = {
            **common,
            "normalization_uid": uid,
            "protocol_id": identity["protocol_id"],
            "cohort_sha256": identity["cohort_sha256"],
            "array_file": "data/capture/"
            + str((folder / "capture_normalization.npz").relative_to(output)),
            "metadata_file": "data/capture/"
            + str((folder / "capture_normalization.json").relative_to(output)),
            "token_count": identity["token_count"],
            "prompt_count": metadata["prompt_count"],
            "feature_count": metadata["feature_count"],
            "valid_feature_count": metadata["valid_feature_count"],
        }
    if not ledger.exists():
        return report, normalization, None
    annotation_snapshot = json.loads(supplement.read_text()) if supplement.exists() else {}
    annotation_export = annotation_snapshot.get("annotation", {}).get("export_sha256")
    partition = Path(expected_mode) / record["model"] / (record["dataset"] + ".parquet")
    activations = Rows(output / "activations" / partition, types["activations"])
    samples = []
    try:
        for row in read_samples(root):
            result = row.get("result")
            if result is None:
                continue
            if _mode(result) != expected_mode or "tokens" in result:
                raise ValueError("Select final-token records for the declared capture mode")
            identity = result["input_identity"]
            site = {
                "model_key": result["model_key"],
                "sae_key": result["sae_key"],
                "model_sha256": identity["model_sha256"],
                "sae_sha256": identity["sae_sha256"],
                "layer": result["layer"],
                "hook_name": result["site"],
            }
            site["site_uid"] = digest(site)
            report.update(site)
            sample_uid = digest(
                {
                    "dataset": record["dataset"],
                    "sample_id": row["sample_id"],
                    "prompt": result["prompt_sha256"],
                }
            )
            selected = {
                int(f["feature_id"]): i + 1 for i, f in enumerate(result.get("top_features", []))
            }
            active = {
                int(f["feature_id"]): i + 1 for i, f in enumerate(result.get("active_features", []))
            }
            merged = {int(f["feature_id"]): f for f in result.get("active_features", [])}
            merged.update({int(f["feature_id"]): f for f in result.get("top_features", [])})
            annotation = result.get("neuronpedia") or {}
            export_sha = annotation.get("export_sha256") or annotation_export
            for index, feature in merged.items():
                uid = _add_feature(features, site, feature, export_sha)
                activations.append(
                    {
                        **common,
                        "sample_uid": sample_uid,
                        "feature_uid": uid,
                        "feature_id": index,
                        "token_position": -1,
                        "activation": feature["activation"],
                        "zscore": feature.get("zscore"),
                        "mean": feature.get("mean"),
                        "std": feature.get("std"),
                        "is_top": index in selected,
                        "is_active": index in active,
                        "top_rank": selected.get(index),
                        "active_rank": active.get(index),
                        "zero_native_activation": feature.get(
                            "zero_native_activation", feature["activation"] == 0
                        ),
                        "whiten_eligible": feature.get("whiten_eligible"),
                    }
                )
            samples.append(
                {
                    **common,
                    "sample_uid": sample_uid,
                    "sample_id": row["sample_id"],
                    "source_split": row.get("split"),
                    "source_index": str(row.get("source_index", "")),
                    "prompt_sha256": result["prompt_sha256"],
                    "source_status": row.get("status"),
                    "token_position": -1,
                    "active_count": len(active),
                    "top_count": len(selected),
                    "top_feature_uids": [site["site_uid"] + ":" + str(index) for index in selected],
                }
            )
    finally:
        activations.close()
    report["sample_count"] = len(samples)
    report["observation_count"] = activations.count
    return report, normalization, (partition, samples)


def _matrix_context(path: Path, mode: str) -> tuple[Path, str]:
    from saebase.runtime.experiment_paths import capture_method_role, matrix_report

    absolute = path.resolve()
    for root in absolute.parents:
        if (root / "experiment" / "batches" / "capture") not in absolute.parents:
            continue
        relative = absolute.relative_to(root / "experiment" / "batches" / "capture")
        name = relative.parts[0]
        expected = matrix_report(root, "capture", name, capture_method_role(name, mode))
        if absolute != expected / "summary.json":
            raise ValueError(f"Expected capture summary at {expected / 'summary.json'}")
        return root, name
    raise ValueError(f"Expected a capture summary at its standard batch path: {path}")


def export_matrices(selections, output):
    """Create a dataset snapshot from selected capture matrices and saved explanations."""
    from saebase.runtime.experiment_paths import capture_method_role, member_directory
    from saebase.runtime.experiment_results import historical_run_path

    matrices = []
    for mode, path in selections:
        source = historical_run_path(path)
        matrices.append((mode, source, *_matrix_context(source, mode)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    types = schemas()
    features, runs, normalizations, sample_tables = {}, [], [], []
    source_matrices = []
    for mode, path, root, name in matrices:
        matrix = json.loads(path.read_text())
        source_matrices.append(
            {"mode": mode, "name": name, "sha256": file_digest(path)}
        )
        for record in matrix["records"]:
            folder = member_directory(
                root, record["dataset"], record["model"], "capture", name,
                capture_method_role(name, mode),
            )
            if not (folder / "metadata/manifest.json").is_file():
                if record.get("tasks", {}).get("capture"):
                    raise FileNotFoundError(f"Capture member is missing its manifest: {folder}")
                continue
            report, normalization, samples = _run(folder, record, mode, output, types, features, root)
            runs.append(report)
            if normalization:
                normalizations.append(normalization)
            if samples:
                sample_tables.append((report, *samples))
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "mode",
                            "model_key",
                            "dataset",
                            "sample_count",
                            "observation_count",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    covered = set()
    dictionaries = {}
    for uid, feature in features.items():
        explanations = list(feature.pop("_explanations").values())
        statuses = sorted(feature.pop("_statuses"))
        feature["annotation_exports"] = sorted(feature.pop("_exports"))
        feature["observed_statuses"] = statuses
        feature["explanations"] = explanations
        feature["explanation_count"] = len(explanations)
        feature["annotation_status"] = "available" if explanations else statuses[0]
        if explanations:
            covered.add(uid)
        dictionaries.setdefault(feature["site_uid"], []).append(feature)
    for site_uid, values in dictionaries.items():
        write_rows(
            output / "features" / (site_uid + ".parquet"),
            sorted(values, key=lambda r: r["feature_id"]),
            types["features"],
        )
    for report, partition, samples in sample_tables:
        for row in samples:
            row["top_explained_count"] = sum(uid in covered for uid in row["top_feature_uids"])
            row["top_features_explained"] = row["top_explained_count"] == row["top_count"]
        report["top_explained_sample_count"] = sum(r["top_features_explained"] for r in samples)
        report["annotation_status"] = (
            "complete" if report["top_explained_sample_count"] == len(samples) else "partial"
        )
        write_rows(output / "samples" / partition, samples, types["samples"])
    write_rows(output / "runs.parquet", runs, types["runs"])
    write_rows(output / "normalization.parquet", normalizations, types["normalization"])
    totals = {}
    for mode in ("non_whiten", "whiten"):
        chosen = [r for r in runs if r["mode"] == mode]
        totals[mode] = {
            "run_slots": len(chosen),
            "captured_runs": sum(r["sample_count"] > 0 for r in chosen),
            "samples": sum(r["sample_count"] for r in chosen),
            "observations": sum(r["observation_count"] for r in chosen),
            "samples_with_top_explanations": sum(r["top_explained_sample_count"] for r in chosen),
        }
    result = {
        "schema": "saebase.capture.parquet.v1",
        "source_matrices": source_matrices,
        "exporter_sha256": file_digest(__file__),
        "totals": totals,
        "features": len(features),
        "explained_features": len(covered),
        "normalizations": len(normalizations),
        "model_keys": sorted({r["model_key"] for r in runs}),
        "datasets": sorted({r["dataset"] for r in runs}),
        "files": {
            p.relative_to(output).as_posix(): {"sha256": file_digest(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", action="append", required=True, help="mode=path/to/summary.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selections = [value.split("=", 1) for value in args.matrix]
    if any(len(v) != 2 or v[0] not in {"non_whiten", "whiten"} for v in selections):
        parser.error("Use non_whiten=path or whiten=path")
    result = export_matrices(selections, args.output)
    print(
        json.dumps(
            {k: result[k] for k in ("totals", "features", "explained_features", "normalizations")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
