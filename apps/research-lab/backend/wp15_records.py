"""Read-only WP15 measurement cards.

These files record numbers from real local runs. They do not contain prompt
text, decoded continuations, explanations, weights, or logits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import core

NAMES = ("benchmark", "training", "generation", "explanation", "circuit")
FORBIDDEN = {
    "description",
    "prompt",
    "prompt_text",
    "continuation",
    "baseline_continuation",
    "edited_continuation",
    "tokens",
    "logits",
    "baseline_logits",
    "edited_logits",
}
SHA64 = set("0123456789abcdef")


def _root() -> Path:
    """Prefer an explicitly staged, read-only release bundle over built-in previews."""
    staged = core.data_root() / "wp15"
    if staged.is_dir():
        return staged
    return Path(__file__).resolve().parents[1] / "data" / "wp15"


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in SHA64 for ch in value)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _reject_secrets(value: Any, *, allow_reviewed_description: bool = False) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN - ({"description"} if allow_reviewed_description else set())
        found = forbidden.intersection(value)
        if found:
            raise ValueError("WP15 card contains unpublished text or weights")
        for item in value.values():
            _reject_secrets(item, allow_reviewed_description=allow_reviewed_description)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item, allow_reviewed_description=allow_reviewed_description)


def load_record(name: str) -> dict[str, Any]:
    if name not in NAMES:
        raise ValueError("Unknown WP15 record")
    path = _root() / f"{name}.json"
    if not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError("WP15 record missing or oversized")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("WP15 record must be an object")
    reviewed_description = (
        name == "explanation"
        and record.get("approved") is True
        and record.get("human_reviewed") is True
        and record.get("rights_reviewed") is True
        and record.get("description_published") is True
    )
    _reject_secrets(record, allow_reviewed_description=reviewed_description)
    if type(record.get("rights_reviewed")) is not bool:
        raise ValueError("WP15 cards require explicit publication review state")
    if name == "benchmark":
        metrics = record.get("metrics")
        if (
            record.get("comparable") is not False
            or record.get("upstream_commit") != "8042bb3828c6340da8d12062324e92b2077c571c"
            or not _sha(record.get("evidence_sha256"))
            or not isinstance(metrics, dict)
            or set(metrics)
            != {
                "saebench.core.l0",
                "saebench.core.mse",
                "saebench.sparse_probing.sae_top_1_test_accuracy",
            }
            or not all(_finite(v) and v >= 0 for v in metrics.values())
        ):
            raise ValueError("Native benchmark card does not match the pinned evaluator")
    elif name == "training":
        if (
            record.get("native_execution") is not True
            or record.get("finite") is not True
            or record.get("weights_published") is not False
            or not _sha(record.get("checkpoint_sha256"))
            or not _sha(record.get("dataset_revision"))
            or record.get("steps") != 50
            or not _finite(record.get("reconstruction_mse"))
            or record["reconstruction_mse"] < 0
        ):
            raise ValueError("Training card is missing a real reloaded checkpoint")
    elif name == "generation":
        samples = record.get("samples")
        if (
            record.get("public_prompt") is not False
            or not isinstance(samples, list)
            or len(samples) != 2
        ):
            raise ValueError("Generation card must keep both prompts private")
        for sample in samples:
            if not _sha(sample.get("prompt_sha256")) or type(sample.get("feature_id")) is not int:
                raise ValueError("Generation sample is missing its source identity")
            if not isinstance(sample.get("conditions"), list) or len(sample["conditions"]) != 3:
                raise ValueError("Generation sample needs noop, ablation, and scale")
    elif name == "explanation":
        if type(record.get("feature_id")) is not int or not _finite(
            record.get("heldout_score")
        ):
            raise ValueError("Explanation card has no measured feature or held-out result")
        if reviewed_description:
            description = record.get("description")
            if (
                record.get("private_unpublished") is not False
                or not isinstance(description, str)
                or not 1 <= len(description.strip()) <= 600
            ):
                raise ValueError("Approved explanation must contain reviewed display text")
        elif (
            record.get("approved") is not False
            or record.get("private_unpublished") is not True
            or record.get("description_published") is not False
        ):
            raise ValueError("Unreviewed explanation cannot publish a description")
    else:
        graph = record.get("graph")
        if (
            type(record.get("rights_reviewed")) is not bool
            or record.get("public_demo_prompt") is not True
            or not isinstance(graph, dict)
            or graph.get("schema") != "opensae.wp12.p6.graph.v1"
            or not isinstance(graph.get("nodes"), list)
            or not isinstance(graph.get("edges"), list)
            or not 1 <= len(graph["nodes"]) <= 250
            or len(graph["edges"]) > 750
            or not any(node.get("kind") == "error" for node in graph["nodes"])
            or not any(node.get("kind") == "logit" for node in graph["nodes"])
            or not _sha((graph.get("provenance") or {}).get("source_sha256"))
        ):
            raise ValueError("Circuit card must be a bounded graph from the real attribution")
    return record
