"""Redacted CPU lexical near-duplicate screening over curated prompt candidates.

This is a deterministic, deliberately conservative *lexical* screen. It does
not certify semantic equivalence, model leakage, downstream validity or
redistribution rights. Only sample identities and overlap scores are emitted.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ingest import _raw_jsonl


def _shingles(text: str, *, width: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if not words:
        return set()
    if len(words) < width:
        return {tuple(words)}
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def scan_near_duplicates(
    frozen_root: Path, dataset_ids: list[str], *, threshold: float = 0.85
) -> dict[str, Any]:
    """Review lexical five-word-shingle similarity without publishing text."""
    if not 0.5 <= threshold <= 1:
        raise ValueError("Threshold outside supported range")
    rows: list[dict[str, Any]] = []
    shingles: list[set[tuple[str, ...]]] = []
    for dataset_id in dataset_ids:
        path = frozen_root / dataset_id / "records.jsonl"
        if not path.exists():
            continue
        for row in _raw_jsonl(path):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": row["sample_id"],
                    "usage_split": row["usage_split"],
                }
            )
            shingles.append(_shingles(row["prompt"]))
    posting: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, values in enumerate(shingles):
        for value in values:
            posting[value].append(i)
    candidate_pairs: set[tuple[int, int]] = set()
    for indices in posting.values():
        if len(indices) > 80:
            # Ignore ubiquitous prompt-template words; they are too common
            # to be useful for detecting near copies of underlying content.
            continue
        candidate_pairs.update(itertools.combinations(sorted(indices), 2))
    matches = []
    for left, right in sorted(candidate_pairs):
        a, b = shingles[left], shingles[right]
        union = len(a | b)
        if not union:
            continue
        similarity = len(a & b) / union
        if similarity >= threshold:
            matches.append(
                {
                    "left_id": rows[left]["sample_id"],
                    "right_id": rows[right]["sample_id"],
                    "similarity": round(similarity, 4),
                    "cross_dataset": rows[left]["dataset_id"] != rows[right]["dataset_id"],
                    "cross_usage_split": rows[left]["usage_split"] != rows[right]["usage_split"],
                }
            )
    return {
        "algorithm": "casefolded_unicode_word_5gram_jaccard_v1",
        "threshold": threshold,
        "candidate_count": len(rows),
        "lexical_near_duplicate_count": len(matches),
        "candidates": matches[:200],
        "truncated": len(matches) > 200,
        "interpretation": "automatic lexical screening only; semantic and human review pending",
    }
