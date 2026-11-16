"""Capture-owned explanation refresh and reading of saved observations.

Reuses the Model/SAE explanation API. No weights or activations are recomputed;
original measurement ledgers stay unchanged. A small metadata update makes the
current completion visible through the existing Runtime result reader.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from saebase.api.metadata import SAEIdentity
from saebase.api.neuronpedia import annotate_features
from saebase.results import completion_summary
from saebase.runtime.experiment_results import write_json
from tasks.capture.operation import neuronpedia_settings


def feature_groups(result):
    """All recorded feature groups, including optional full-token traces."""
    return (
        [result.get("active_features", []), result.get("top_features", [])]
        + [token["features"] for token in result.get("tokens", [])]
        + [
            pair.get("features", [])
            for pair in result.get("sar", {}).get("pairwise", {}).get("top_pairs", [])
        ]
    )


def required_features(result):
    """Features requested by the capture output, whose explanations determine completion."""
    if "tokens" in result:
        return [feature for token in result["tokens"] for feature in token["features"]]
    features = result.get("top_features", []) + [
        feature
        for pair in result.get("sar", {}).get("pairwise", {}).get("top_pairs", [])
        for feature in pair.get("features", [])
    ]
    return list({feature["feature_id"]: feature for feature in features}.values())


def _recorded_identity(root: Path, summary: dict) -> SAEIdentity:
    raw = summary.get("sae_identity")
    if raw is None:
        # Earlier capture runs recorded their loaded identity in precheck.
        receipt = root / "tmp/automatic_precheck/metadata/precheck_result.json"
        checks = json.loads(receipt.read_text())["rows"]
        for check in checks:
            candidates = check.get("samples", [check])
            raw = next((item["identity"] for item in candidates if "identity" in item), None)
            if raw is not None:
                break
    if raw is None:
        raise ValueError(
            "This capture has no recorded loaded SAE identity; supply the original run evidence"
        )
    identity = SAEIdentity(**{f.name: raw[f.name] for f in fields(SAEIdentity) if f.name in raw})
    if (identity.sae_key, identity.layer) != (summary["sae_key"], summary["layer"]):
        raise ValueError("Recorded precheck SAE differs from capture")
    return identity


def refresh_annotations(folder: str | Path) -> dict:
    """Fetch explanations for saved feature IDs and publish a completion update."""
    root = Path(folder).resolve()
    cfg = OmegaConf.load(root / "metadata/config.yaml")
    manifest = json.loads((root / "metadata/manifest.json").read_text())
    if manifest["task"] != "capture":
        raise ValueError("Explanation refresh requires a capture run")
    summary = manifest["result"]["summary"]
    ledger = root / "reports/capture_samples.jsonl"
    content = ledger.read_bytes()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids = {
        feature["feature_id"]
        for row in rows
        if "result" in row
        for group in feature_groups(row["result"])
        for feature in group
    }
    pair = SimpleNamespace(
        identity=_recorded_identity(root, summary), sae=SimpleNamespace(cfg=None)
    )
    info, descriptions = annotate_features(pair, sorted(ids), neuronpedia_settings(cfg))
    annotation_work = []
    work = []
    top_complete = 0
    active_complete = 0
    for row in rows:
        if "result" not in row:
            work.append({"status": row["status"], "reason": row.get("reason", "Capture failed")})
            continue
        result = row["result"]
        selected = {f["feature_id"] for f in required_features(result)}
        complete = info["status"] == "disabled" or all(
            descriptions[i]["status"] == "available" for i in selected
        )
        top_complete += all(
            descriptions[f["feature_id"]]["status"] == "available" for f in result["top_features"]
        )
        active_complete += all(
            descriptions[f["feature_id"]]["status"] == "available"
            for f in result.get("active_features", [])
        )
        item = {"status": "completed" if complete else "blocked"}
        if not complete:
            item["reason"] = (
                info.get("reason") or "Some selected features have no published explanation"
            )
        annotation_work.append(item)
        work.append(
            {"status": "undefined", "reason": "SAR coalition behavior has zero variance"}
            if complete and result.get("sar", {}).get("status") == "undefined"
            else item
        )
    expected = summary.get("planned_count", len(rows))
    overall = completion_summary(work, expected=expected, unit="sample")
    annotation = completion_summary(annotation_work, unit="sample")
    supplement = {
        "schema": "saebase.capture.annotation_supplement.v1",
        "samples_path": "reports/capture_samples.jsonl",
        "samples_sha256": hashlib.sha256(content).hexdigest(),
        "sae_key": summary["sae_key"],
        "layer": summary["layer"],
        "mode": summary.get("mode", "non_whiten"),
        "annotation": info,
        "features": {str(k): v for k, v in descriptions.items()},
        "coverage": {
            "samples": len(rows),
            "samples_with_all_top_features_explained": top_complete,
            "samples_with_all_active_features_explained": active_complete,
        },
        "original_measurements_unchanged": True,
    }
    write_json(root / "reports/capture_explanations.json", supplement)
    update = {
        "completion": overall,
        "annotation": annotation,
        "samples_bytes": len(content),
        "coverage": supplement["coverage"],
        "explanations_path": "reports/capture_explanations.json",
    }
    write_json(root / "metadata/capture_annotation_update.json", update)
    return update


def read_samples(folder: str | Path):
    """Yield saved capture rows with their latest published explanations."""
    root = Path(folder)
    supplement = root / "reports/capture_explanations.json"
    enrichment = json.loads(supplement.read_text()) if supplement.exists() else {}
    descriptions = enrichment.get("features", {})
    required = enrichment.get("annotation", {}).get("status") != "disabled"
    with (root / "reports/capture_samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if "result" in row:
                for group in feature_groups(row["result"]):
                    for feature in group:
                        if str(feature["feature_id"]) in descriptions:
                            feature["neuronpedia"] = descriptions[str(feature["feature_id"])]
                if descriptions:
                    selected = {
                        f["feature_id"]: f["neuronpedia"]
                        for group in feature_groups(row["result"])
                        for f in group
                    }
                    explained = sum(value["status"] == "available" for value in selected.values())
                    row["result"]["neuronpedia"] = {
                        "provider": "neuronpedia",
                        "requested": len(selected),
                        "explained": explained,
                        "status": (
                            "disabled"
                            if not required
                            else "complete"
                            if explained == len(selected)
                            else "partial"
                        ),
                        "source_ref": "reports/capture_explanations.json",
                    }
                    work = [{"status": "completed"}]
                    if row["result"].get("sar", {}).get("status") == "undefined":
                        work.append(
                            {
                                "status": "undefined",
                                "reason": "SAR coalition behavior has zero variance",
                            }
                        )
                    if required:
                        work.extend(
                            {
                                "status": "completed"
                                if feature["neuronpedia"]["status"] == "available"
                                else "blocked"
                            }
                            for feature in required_features(row["result"])
                        )
                    completion = completion_summary(work, unit="work_item")
                    row["result"]["completion"] = completion
                    row["status"] = (
                        "blocked"
                        if completion["status"] != "complete"
                        else "undefined"
                        if row["result"].get("sar", {}).get("status") == "undefined"
                        else "completed"
                    )
            yield row


def read_result(root: Path, manifest: dict) -> dict:
    """Project a small Task-owned annotation update; never rescan the ledger."""
    path = root / "metadata/capture_annotation_update.json"
    if not path.is_file():
        return manifest
    update = json.loads(path.read_text())
    payload = dict(manifest["result"])
    summary = dict(payload["summary"])
    summary.update(
        completion=update["completion"],
        stages={**summary.get("stages", {}), "annotation": update["annotation"]},
        annotation_update=update,
    )
    required = list(summary.get("required_artifacts", []))
    if not any(item["path"] == update["explanations_path"] for item in required):
        required.append({"path": update["explanations_path"]})
    summary["required_artifacts"] = required
    payload.update(
        summary=summary,
        completion=update["completion"],
        ok=manifest["status"] == "executed" and update["completion"]["status"] == "complete",
    )
    return {
        **manifest,
        "result": payload,
        "completion": update["completion"],
        "exit_code": 1 if update["completion"]["status"] == "failed" else 0 if payload["ok"] else 2,
    }
