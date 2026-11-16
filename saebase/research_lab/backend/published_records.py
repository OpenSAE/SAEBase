"""Read approved public records and import source-bound circuit display candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from tasks.circuits.backend.circuits import import_native_graph_file, prune

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
    if "completion" in record:
        completion = record["completion"]
        if not isinstance(completion, dict) or completion.get("status") not in {
            "complete",
            "incomplete",
            "failed",
        }:
            raise ValueError("Invalid task-owned completion summary")
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
            if "completion" in record:
                info["completion"] = record["completion"]
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


DEFAULT_TOOL_COMMIT = "255d989e93be1299d53d67b59e9fd76a0e274b14"


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def import_circuit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        type=Path,
        required=True,
        help="Genuine circuit-tracer frontend v1 graph JSON",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Exact model checkpoint snapshot used during tracing",
    )
    parser.add_argument(
        "--transcoder-dir",
        type=Path,
        required=True,
        help="Exact compatible transcoder snapshot used during tracing",
    )
    parser.add_argument(
        "--card-output",
        type=Path,
        required=True,
        help="New reviewed-candidate circuit card JSON destination",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
        help="New private summary destination",
    )
    parser.add_argument("--tool-commit", default=DEFAULT_TOOL_COMMIT)
    parser.add_argument(
        "--public-demo-prompt",
        action="store_true",
        help="Confirm the original prompt was authored and may be shown",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.tool_commit):
        parser.error("--tool-commit requires the exact 40-character upstream source revision")
    if args.card_output.exists() or args.summary_output.exists():
        parser.error("Do not overwrite original experiment or release files")

    model_files = sorted(args.model_dir.glob("model-*-of-*.safetensors"))
    if not model_files:
        model_files = sorted(args.model_dir.glob("*.safetensors"))
    transcoder_files = sorted(args.transcoder_dir.rglob("layer_*.safetensors"))
    if not model_files or not transcoder_files:
        parser.error(
            "Missing checkpoint files in the exact user-selected model or transcoder snapshot: "
            f"model={len(model_files)}, transcoder={len(transcoder_files)}"
        )
    record = import_native_graph_file(
        args.native,
        model_sha256=_hash_files(model_files),
        transcoder_sha256=_hash_files(transcoder_files),
        original_tool_commit=args.tool_commit,
    )
    thresholds = sorted({abs(edge.weight) for edge in record.edges}, reverse=True)
    selected = None
    for threshold in thresholds:
        candidate = prune(record, min_abs_weight=threshold)
        if len(candidate.nodes) <= 250 and len(candidate.edges) <= 750:
            selected = candidate
        else:
            break
    if selected is None:
        raise SystemExit("No pruned graph fits the browser limit")
    card = {
        "schema": "opensae.wp15.circuit_record.v1",
        "rights_reviewed": False,
        "public_demo_prompt": args.public_demo_prompt,
        "source": "mwhanna/gemma-scope-transcoders",
        "model": "google/gemma-2-2b",
        "tool_commit": args.tool_commit,
        "full_node_count": len(record.nodes),
        "full_edge_count": len(record.edges),
        "graph": selected.payload(),
    }
    args.card_output.parent.mkdir(parents=True, exist_ok=True)
    # Compact so the bounded graph stays under the Lab card size cap.
    args.card_output.write_text(json.dumps(card, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "full_nodes": len(record.nodes),
        "full_edges": len(record.edges),
        "shown_nodes": len(selected.nodes),
        "shown_edges": len(selected.edges),
        "model_files": len(model_files),
        "transcoder_files": len(transcoder_files),
        "source_sha256": selected.source_sha256,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    raise SystemExit(import_circuit_main())
