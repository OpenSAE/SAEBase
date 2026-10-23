#!/usr/bin/env python3
"""Import the real Gemma-2-2B circuit-tracer graph and keep a browser-sized copy.

The raw .pt file stays on disk. This script reads the official frontend JSON,
checks node identities, and keeps the strongest edges that fit the Lab view.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
from pathlib import Path

from saebase.wp12.circuits import import_native_graph_file, prune

DEFAULT_TOOL_COMMIT = "255d989e93be1299d53d67b59e9fd76a0e274b14"


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True,
                        help="Genuine circuit-tracer frontend v1 graph JSON")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Exact model checkpoint snapshot used during tracing")
    parser.add_argument("--transcoder-dir", type=Path, required=True,
                        help="Exact compatible transcoder snapshot used during tracing")
    parser.add_argument("--card-output", type=Path, required=True,
                        help="New reviewed-candidate circuit card JSON destination")
    parser.add_argument("--summary-output", type=Path, required=True,
                        help="New private summary destination")
    parser.add_argument("--tool-commit", default=DEFAULT_TOOL_COMMIT)
    parser.add_argument("--public-demo-prompt", action="store_true",
                        help="Confirm the original prompt was authored and may be shown")
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
    main()
