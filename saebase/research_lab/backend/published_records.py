"""Immutable, read-only published research experiment records.

Only owner-approved nonsecret artifacts are served. Private original prompts,
training weights, job errors and credential-bearing logs are never exposed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from . import core

CATEGORIES = {"interventions", "benchmarks", "training", "circuits"}
SCHEMAS = {
    "interventions": "opensae.wp12.p3.public_result.v1",
    "benchmarks": "opensae.wp12.p4.comparison.v1",
    "training": "opensae.wp12.p5.summary.v1",
    "circuits": "opensae.wp12.p6.graph.v1",
}
MAX_BYTES = 2 * 1024 * 1024
MAX_ITEMS = 100


def _root(root: Path | None = None) -> Path:
    return (root or core.data_root()) / "published_wp12"


def _id(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _validate(category: str, record: dict[str, Any]) -> None:
    if category not in CATEGORIES or not isinstance(record, dict):
        raise ValueError("Unsupported published research record category")
    if record.get("schema") != SCHEMAS[category]:
        raise ValueError("Wrong typed public record schema")
    if category == "circuits":
        nodes = record.get("nodes")
        edges = record.get("edges")
        info = record.get("provenance", {})
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Graph requires actual node/edge arrays")
        if not 1 <= len(nodes) <= 250 or len(edges) > 750:
            raise ValueError("Browser graph needs a source-bound bounded subgraph")
        node_ids = set()
        for node in nodes:
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("id"), str)
                or node.get("kind") not in {"feature", "token", "error", "logit"}
            ):
                raise ValueError("Invalid public circuit node")
            node_ids.add(node["id"])
        if len(node_ids) != len(nodes) or any(
            not isinstance(edge, dict)
            or edge.get("source") not in node_ids
            or edge.get("target") not in node_ids
            or type(edge.get("weight")) not in (int, float)
            for edge in edges
        ):
            raise ValueError("Invalid public graph edge identity")
        if not isinstance(info, dict) or not _id(info.get("source_sha256")):
            raise ValueError("Original signed attribution source hash missing")
        if not any(n["kind"] == "error" for n in nodes):
            raise ValueError("Original error nodes must remain present")
    elif category == "benchmarks":
        if record.get("comparable") is not True or len(record.get("points", [])) < 2:
            raise ValueError("Only actual fair matched-source comparisons are published")
    elif category == "training":
        if not record.get("native_execution") or not record.get("checkpoint_sha256"):
            raise ValueError("Training cards require actual native source and checkpoint")
    else:
        if not all(_id(record.get(k)) for k in ("model_sha256", "sae_sha256")):
            raise ValueError("Intervention cards require real model/SAE weight identities")
        if record.get("measured") is not True:
            raise ValueError("Do not publish an unexecuted intervention as a result")


def publish_record(
    category: str,
    record: dict[str, Any],
    *,
    root: Path | None = None,
    public_consent: bool = False,
    rights_reviewed: bool = False,
) -> dict[str, Any]:
    if not public_consent or not rights_reviewed:
        raise PermissionError("Explicit reviewed permission required for public publication")
    _validate(category, record)
    payload = core.canonical_json(record)
    if len(payload) > MAX_BYTES:
        raise ValueError("Public research record exceeds the browser limit")
    sha = hashlib.sha256(payload).hexdigest()
    folder = _root(root) / category / sha
    if folder.exists():
        raise FileExistsError("Published records cannot be changed in place")
    folder.mkdir(parents=True, mode=0o700)
    file = folder / "record.json"
    file.write_bytes(payload)
    os.chmod(file, 0o600)
    meta = {
        "schema": "opensae.wp12.published.v1",
        "category": category,
        "record_sha256": sha,
        "source_reviewed": True,
        "public_consent": True,
        "rights_reviewed": True,
    }
    (folder / "manifest.json").write_bytes(core.canonical_json(meta))
    return meta


def load_record(
    category: str,
    record_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    if category not in CATEGORIES or not _id(record_id):
        return None
    folder = _root(root) / category / record_id
    if not folder.is_dir():
        return None
    record_path = folder / "record.json"
    manifest_path = folder / "manifest.json"
    if (
        not record_path.is_file()
        or record_path.stat().st_size > MAX_BYTES
        or not manifest_path.is_file()
    ):
        raise core.InvalidEvidence("Missing or oversized public WP12 record")
    meta = core.read_json(manifest_path)
    if (
        meta.get("schema") != "opensae.wp12.published.v1"
        or meta.get("record_sha256") != record_id
        or meta.get("category") != category
        or meta.get("public_consent") is not True
        or meta.get("rights_reviewed") is not True
        or core.digest(record_path) != record_id
    ):
        raise core.InvalidEvidence("Changed public WP12 manifest or source record")
    record = core.read_json(record_path)
    try:
        _validate(category, record)
    except (ValueError, TypeError) as exc:
        raise core.InvalidEvidence("Invalid signed public record schema") from exc
    return record


def list_records(
    category: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ValueError("Invalid research collection")
    folder = _root(root) / category
    if not folder.is_dir():
        return {"category": category, "items": [], "total": 0, "mode": "published_only"}
    ids = sorted((x.name for x in folder.iterdir() if x.is_dir() and _id(x.name)))[:MAX_ITEMS]
    items = []
    for name in ids:
        record = load_record(category, name, root=root)
        if record is not None:
            info = {
                "id": name,
                "schema": record["schema"],
                "category": category,
                "source": "owner_approved_real_record",
            }
            if category == "circuits":
                info["nodes"] = len(record["nodes"])
                info["edges"] = len(record["edges"])
            elif category == "benchmarks":
                info["metric"] = record.get("metric")
                info["points"] = len(record["points"])
            elif category == "training":
                info["backend"] = record.get("backend")
            else:
                info["site"] = record.get("site")
            items.append(info)
    return {"category": category, "items": items, "total": len(items), "mode": "published_only"}
