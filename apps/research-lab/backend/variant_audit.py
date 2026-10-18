"""Summarize RSS variant identities on CPU without publishing source text."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .core import canonical_json
from .ingest import _raw_jsonl
from .transforms import build_variants


def summarize_variant_proposals(frozen_root: Path, source_ids: list[str]) -> dict[str, Any]:
    """Make 45 unreviewed variants per frozen prompt and return hashes only."""
    records = []
    for source in source_ids:
        file = frozen_root / source / "records.jsonl"
        if not file.is_file():
            continue
        for row in _raw_jsonl(file):
            proposed = build_variants(
                source,
                row["sample_id"],
                row["prompt_group_id"],
                row["prompt"],
                anchor=row["prompt"],
                source_revision=row["source_revision"],
            )
            if proposed["fully_reviewed"]:
                raise ValueError("Automatically generated variants cannot be certified")
            selection = sum(v["variant_role"] == "selection" for v in proposed["variants"])
            held = sum(v["variant_role"] == "held_out" for v in proposed["variants"])
            if selection != 15 or held != 30 or proposed["n_proposals"] != 45:
                raise ValueError("Incomplete 45-variant proposal manifest")
            content_hashes = [
                {
                    "id": variant["variant_id"],
                    "sha256": hashlib.sha256(variant["text"].encode("utf-8")).hexdigest(),
                    "role": variant["variant_role"],
                }
                for variant in proposed["variants"]
            ]
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_group_id": row["prompt_group_id"],
                    "variants_sha256": hashlib.sha256(canonical_json(content_hashes)).hexdigest(),
                    "selection": selection,
                    "held_out": held,
                    "total": proposed["n_proposals"],
                    "transform_family": proposed["transform_family"],
                }
            )
    return {
        "protocol": "opensae.rss.variant_proposals.v1",
        "parent_groups": len(records),
        "proposed_variants": sum(row["total"] for row in records),
        "selection_proposals": sum(row["selection"] for row in records),
        "held_out_proposals": sum(row["held_out"] for row in records),
        "metadata_only_sha256": hashlib.sha256(canonical_json(records)).hexdigest(),
        "semantic_review": "pending",
        "tokenizer_alignment": "pending",
        "input_texts_in_audit": False,
    }
