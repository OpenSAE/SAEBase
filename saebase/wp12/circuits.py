"""P6 native circuit-tracer CLI wrapper and typed graph data preservation.

Native attribution must actually run upstream. Pruning, node annotations and
a drawn graph cannot stand in for the gradient computation or intervention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Literal


SHA = re.compile(r"^[0-9a-f]{40}$")
KIND = {"sae", "transcoder", "skip_transcoder", "crosscoder", "crosslayer"}


@dataclass(frozen=True)
class RepresentationAsset:
    kind: Literal["sae", "transcoder", "skip_transcoder", "crosscoder", "crosslayer"]
    repo_id: str
    revision: str
    checkpoint_id: str
    input_hook: str
    output_hook: str
    d_in: int
    d_out: int

    def __post_init__(self) -> None:
        if self.kind not in KIND or not SHA.fullmatch(self.revision):
            raise ValueError("Typed asset and pinned source commit required")
        if not self.repo_id or not self.checkpoint_id or not self.input_hook or not self.output_hook:
            raise ValueError("Exact asset and distinct input/output site metadata required")
        if self.d_in <= 0 or self.d_out <= 0:
            raise ValueError("Representation dimensions must be positive")
        if self.kind == "sae" and self.input_hook != self.output_hook:
            raise ValueError("Same-site SAE cannot be relabeled a cross-layer representation")
        if self.kind == "crosslayer" and self.input_hook == self.output_hook:
            raise ValueError("Cross-layer asset requires distinct input/output hooks")


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: Literal["token", "feature", "error", "logit"]
    layer: int | None
    feature_id: int | None
    token_position: int | None
    label: str

    def __post_init__(self) -> None:
        if not self.id or len(self.id) > 120 or self.kind not in {"token", "feature", "error", "logit"}:
            raise ValueError("Invalid attribution graph node")
        if len(self.label) > 256:
            raise ValueError("Unsafe or unbounded graph label")
        for value in (self.layer, self.feature_id, self.token_position):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Graph indices must be nonnegative")


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("Graph edge must join two distinct nodes")
        if type(self.weight) not in (int, float) or not math.isfinite(self.weight):
            raise ValueError("Signed attribution weight must be finite")


@dataclass(frozen=True)
class GraphRecord:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    source_sha256: str
    model_sha256: str
    transcoder_sha256: str
    original_tool_commit: str
    source_format: str

    def validate(self) -> None:
        ids = [n.id for n in self.nodes]
        if not ids or len(ids) != len(set(ids)) or len(ids) > 3000 or len(self.edges) > 15000:
            raise ValueError("Graph sizes or duplicate node IDs invalid")
        for value in (self.source_sha256, self.model_sha256, self.transcoder_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Full original source/model/transcoder SHA256 required")
        if not SHA.fullmatch(self.original_tool_commit):
            raise ValueError("Native circuit-tracer commit required")
        valid = set(ids)
        if any(e.source not in valid or e.target not in valid for e in self.edges):
            raise ValueError("Dangling attribution edge")
        if not any(n.kind == "error" for n in self.nodes):
            raise ValueError("Original attribution error nodes cannot be dropped")
        if not any(n.kind == "logit" for n in self.nodes):
            raise ValueError("A target logit node must be included")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {"schema": "opensae.wp12.p6.graph.v1",
                "nodes": [asdict(x) for x in self.nodes],
                "edges": [asdict(x) for x in self.edges],
                "provenance": {
                    "source_sha256": self.source_sha256,
                    "model_sha256": self.model_sha256,
                    "transcoder_sha256": self.transcoder_sha256,
                    "original_tool_commit": self.original_tool_commit,
                    "source_format": self.source_format,
                }}


def parse_typed_graph(raw: dict[str, Any], *, provenance: dict[str, str]) -> GraphRecord:
    """Import a verified *normalized* graph; never guess native tensor indexing.

    Upstream circuit-tracer emits raw .pt and its own visualization JSON
    schema. A separate explicit converter must map that exact pinned schema
    to nodes/edges before this function is called. Arbitrary .pt is never
    loaded in the public API.
    """
    if set(raw) != {"nodes", "edges"} or not isinstance(raw["nodes"], list) or not isinstance(raw["edges"], list):
        raise ValueError("Expected explicit normalized nodes/edges, not an unverified native schema")
    record = GraphRecord(
        tuple(GraphNode(**n) for n in raw["nodes"]),
        tuple(GraphEdge(**e) for e in raw["edges"]),
        source_sha256=provenance["source_sha256"],
        model_sha256=provenance["model_sha256"],
        transcoder_sha256=provenance["transcoder_sha256"],
        original_tool_commit=provenance["original_tool_commit"],
        source_format=provenance["source_format"],
    )
    record.validate()
    return record


def prune(record: GraphRecord, *, min_abs_weight: float) -> GraphRecord:
    """Return bounded view without deleting raw graph; retain all error/target nodes."""
    record.validate()
    if type(min_abs_weight) not in (int, float) or not math.isfinite(min_abs_weight) or min_abs_weight < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    edges = tuple(edge for edge in record.edges if abs(edge.weight) >= min_abs_weight)
    keep = {n.id for n in record.nodes if n.kind in {"error", "logit", "token"}}
    keep |= {edge.source for edge in edges} | {edge.target for edge in edges}
    subset = GraphRecord(
        nodes=tuple(n for n in record.nodes if n.id in keep),
        edges=edges, source_sha256=record.source_sha256,
        model_sha256=record.model_sha256,
        transcoder_sha256=record.transcoder_sha256,
        original_tool_commit=record.original_tool_commit,
        source_format=record.source_format,
    )
    subset.validate()
    return subset


def save_graph(record: GraphRecord, destination: Path) -> dict:
    """Save immutable graph JSON and checksum, no downloaded weights or prompts."""
    if destination.exists():
        raise FileExistsError("Cannot replace an existing graph artifact")
    destination.mkdir(parents=True, mode=0o700)
    data = json.dumps(record.payload(), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    sha = hashlib.sha256(data).hexdigest()
    (destination / "graph.json").write_bytes(data)
    manifest = {"schema": "opensae.wp12.p6.graph_manifest.v1",
                "graph_sha256": sha, "node_count": len(record.nodes),
                "edge_count": len(record.edges), "visibility": "private",
                "public_release_approved": False}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def graph_feature_intervention_request(
    record: GraphRecord, node_id: str, asset: RepresentationAsset, *,
    value: float,
) -> dict:
    """Typed intervention request only; actual causal edit runs in native backend."""
    record.validate()
    node = next((n for n in record.nodes if n.id == node_id), None)
    if node is None or node.kind != "feature" or node.feature_id is None:
        raise ValueError("Select a real feature node")
    if asset.kind not in {"transcoder", "skip_transcoder", "crosslayer"}:
        raise ValueError("Graph feature requires its compatible transcoder representation")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Nonfinite target feature value")
    return {"status": "requires_native_feature_intervention",
            "node_id": node.id, "feature_id": node.feature_id,
            "target_value": float(value), "asset": asdict(asset),
            "graph_sha256": hashlib.sha256(json.dumps(record.payload(), sort_keys=True).encode()).hexdigest(),
            "executed": False}


def run_native_circuit_tracer(*, prompt: str, transcoder_set: str,
                              output: Path, slug: str, approved: bool = False,
                              timeout_seconds: int = 1800) -> dict:
    """Use original circuit-tracer CLI, without starting its web server."""
    if not approved:
        raise PermissionError("GPU circuit attribution requires explicit authorization")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 2000:
        raise ValueError("Short explicit prompt required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", slug):
        raise ValueError("Unsafe circuit experiment slug")
    if not (transcoder_set in {"gemma", "llama"} or
            (re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}", transcoder_set)) or
            Path(transcoder_set).is_dir()):
        raise ValueError("Transcoder source must be an approved native shortcut, pinned repo, or local directory")
    if not 30 <= timeout_seconds <= 7200:
        raise ValueError("C1 GPU time budget required")
    if output.exists():
        raise FileExistsError("Do not overwrite an existing original graph")
    if shutil.which("circuit-tracer") is None:
        raise RuntimeError("Install the original circuit-tracer worker in an isolated environment")
    output.mkdir(parents=True, mode=0o700)
    raw = output / "native_graph.pt"
    cmd = ["circuit-tracer", "attribute", "--prompt", prompt,
           "--transcoder_set", transcoder_set, "--slug", slug,
           "--graph_file_dir", str(output / "native_graph_files"),
           "--graph_output_path", str(raw)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_seconds, check=False)
    (output / "native.stdout.log").write_text(proc.stdout)
    (output / "native.stderr.log").write_text(proc.stderr)
    if proc.returncode or not raw.exists():
        raise RuntimeError("Original upstream attribution failed; see source logs")
    return {"status": "native_raw_graph_generated_not_normalized",
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "visualization_directory": str(output / "native_graph_files"),
            "native_json_requires_source_specific_converter": True}
