#!/usr/bin/env python3
"""Import the real Gemma-2-2B circuit-tracer graph and keep a browser-sized copy.

The raw .pt file stays on disk. This script reads the official frontend JSON,
checks node identities, and keeps the strongest edges that fit the Lab view.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from saebase.wp12.circuits import import_native_graph_file, prune

NATIVE = Path("/workspace/wp15/runs/G4-gemma/native_graph_files/wp15-gemma-capital.json")
MODEL = Path(
    "/workspace/.hf_home/hub/models--google--gemma-2-2b/snapshots/"
    "c5ebcd40d208330abc697524c919956e692655cf"
)
TRANSCODER = Path(
    "/workspace/.hf_home/hub/models--mwhanna--gemma-scope-transcoders/snapshots/"
    "bd5773156dea09893636c801df1237d0410307d2"
)
CARD = Path("/workspace/SAEBase-wp15/apps/research-lab/data/wp15/circuit.json")
SUMMARY = Path("/workspace/wp15/runs/G4-gemma/import_summary.json")
TOOL_COMMIT = "255d989e93be1299d53d67b59e9fd76a0e274b14"


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _find(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def main() -> None:
    cache = Path("/workspace/.hf_home")
    model_files = sorted(MODEL.glob("model-*-of-*.safetensors"))
    if not model_files:
        model_files = _find(cache / "hub", "model-*-of-*.safetensors")
    transcoder_files = _find(TRANSCODER, "layer_*.safetensors")
    if not transcoder_files:
        transcoder_files = _find(cache / "hub", "layer_*.safetensors")
        transcoder_files = [path for path in transcoder_files if "gemma" in str(path)]
    if not model_files or not transcoder_files:
        raise SystemExit(
            f"Missing weights: model={len(model_files)} transcoder={len(transcoder_files)}"
        )
    record = import_native_graph_file(
        NATIVE,
        model_sha256=_hash_files(model_files),
        transcoder_sha256=_hash_files(transcoder_files),
        original_tool_commit=TOOL_COMMIT,
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
        "public_demo_prompt": True,
        "source": "mwhanna/gemma-scope-transcoders",
        "model": "google/gemma-2-2b",
        "tool_commit": TOOL_COMMIT,
        "full_node_count": len(record.nodes),
        "full_edge_count": len(record.edges),
        "graph": selected.payload(),
    }
    CARD.parent.mkdir(parents=True, exist_ok=True)
    # Compact so the bounded graph stays under the Lab card size cap.
    CARD.write_text(json.dumps(card, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "full_nodes": len(record.nodes),
        "full_edges": len(record.edges),
        "shown_nodes": len(selected.nodes),
        "shown_edges": len(selected.edges),
        "model_files": len(model_files),
        "transcoder_files": len(transcoder_files),
        "source_sha256": selected.source_sha256,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
