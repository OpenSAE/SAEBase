"""Fail-closed verification of the frozen *unreviewed* 10 x 50 corpus.

The lock stores no third-party prompt text. It freezes 10 upstream commits,
the deterministic source-export digests, 50 rendered-candidate digests per
source, and the whole redacted 500-row audit fingerprint. It does NOT
certify rights to republish, scientific quality, or absence of near duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .core import PLATFORM, canonical_json, read_json

LOCK_PATH = PLATFORM / "data" / "source_lock_v1.json"
SCHEMA = "opensae.corpus_source_lock.v1"


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_corpus_lock(
    report: dict[str, Any],
    lock: dict[str, Any] | None = None,
    *,
    verify_aggregate: bool = True,
) -> dict[str, Any]:
    """Reject a changed candidate list even if counts remain 10 x 50."""
    anchor = read_json(LOCK_PATH) if lock is None else lock
    if anchor.get("schema") != SCHEMA:
        raise ValueError("Wrong corpus lock schema")
    if report.get("schema") != "opensae.corpus_cpu_audit.v1":
        raise ValueError("Wrong metadata-only source audit schema")
    if (
        report.get("sources_selected") != 10
        or report.get("candidate_count") != 500
        or report.get("number_sources_target") != 10
        or report.get("gpu_calls") != 0
        or report.get("full_source_text_in_this_artifact") is not False
        or report.get("cross_source_exact_duplicates") != []
        or len(report.get("dataset_results", [])) != 10
        or anchor.get("total_sources") != 10
        or anchor.get("total_base_groups") != 500
        or anchor.get("discovery_groups") != 200
        or anchor.get("evaluation_groups") != 300
    ):
        raise ValueError("Incorrect 10 x 50 unreviewed source population or duplicate report")
    if anchor.get("rights_review") != "pending" or anchor.get("human_content_review") != "pending":
        raise ValueError("This metadata lock cannot certify human/rights approval")
    entries = anchor.get("sources")
    if not isinstance(entries, list) or len(entries) != 10:
        raise ValueError("Expected exactly ten pinned source entries")
    expected = {row["dataset_id"]: row for row in entries}
    if len(expected) != 10:
        raise ValueError("Duplicate locked dataset ID")
    actual = {row.get("dataset_id"): row for row in report["dataset_results"]}
    if len(actual) != 10 or set(actual) != set(expected):
        raise ValueError("Selected source identities do not match lock")
    all_sample_ids, all_groups, all_text_sha = set(), set(), set()
    for dataset_id, ref in expected.items():
        data = actual[dataset_id]
        if data.get("status") != "candidate_frozen_unreviewed":
            raise ValueError("Unfrozen dataset: " + dataset_id)
        for field in ("source_revision", "source_export_sha256", "candidate_sha256"):
            if data.get(field) != ref.get(field):
                raise ValueError("Pinned data identity mismatch: " + dataset_id + " " + field)
        if (
            data.get("max_raw_rows") != ref.get("max_raw_rows")
            or data.get("n_candidates") != 50
            or data.get("discovery") != 20
            or data.get("evaluation") != 30
            or data.get("license_reviewed") is not False
            or data.get("human_reviewed") is not False
        ):
            raise ValueError("Candidate selection or review gate changed: " + dataset_id)
        rows = data.get("candidate_rows_without_text")
        if not isinstance(rows, list) or len(rows) != 50:
            raise ValueError("Missing redacted sample identity rows: " + dataset_id)
        usage = {"discovery": 0, "evaluation": 0}
        for row in rows:
            if (
                row.get("dataset_id") != dataset_id
                or row.get("source_revision") != ref["source_revision"]
            ):
                raise ValueError("Incorrect row origin: " + dataset_id)
            if "prompt" in row or "text" in row:
                raise ValueError("Redacted lock unexpectedly contains source prompt text")
            if (
                not _is_sha(row.get("text_sha256"))
                or not isinstance(row.get("sample_id"), str)
                or not row["sample_id"].startswith(dataset_id + ":")
                or not isinstance(row.get("prompt_group_id"), str)
                or not row["prompt_group_id"].startswith(dataset_id + ":")
                or row.get("usage_split") not in usage
            ):
                raise ValueError("Malformed redacted candidate identity")
            usage[row["usage_split"]] += 1
            for seen, key in (
                (all_sample_ids, "sample_id"),
                (all_groups, "prompt_group_id"),
                (all_text_sha, "text_sha256"),
            ):
                value = row[key]
                if value in seen:
                    raise ValueError("Repeated source sample, group or exact content hash")
                seen.add(value)
        if usage != {"discovery": 20, "evaluation": 30}:
            raise ValueError("Incorrect source usage split: " + dataset_id)
    if anchor.get("rss_variant_proposals_target") is not None:
        variants = report.get("rss_variant_proposals")
        if (
            not isinstance(variants, dict)
            or variants.get("protocol") != "opensae.rss.variant_proposals.v1"
            or variants.get("parent_groups") != 500
            or variants.get("proposed_variants") != anchor["rss_variant_proposals_target"]
            or variants.get("selection_proposals") != anchor.get("rss_selection_proposals")
            or variants.get("held_out_proposals") != anchor.get("rss_held_out_proposals")
            or variants.get("metadata_only_sha256")
            != anchor.get("rss_variant_proposals_metadata_sha256")
            or variants.get("input_texts_in_audit") is not False
            or variants.get("semantic_review") != "pending"
            or variants.get("tokenizer_alignment") != "pending"
        ):
            raise ValueError("RSS variant population not identical to frozen candidate lock")
    if anchor.get("lexical_near_duplicate_count") is not None:
        screened = report.get("lexical_near_duplicate_screen")
        if (
            not isinstance(screened, dict)
            or screened.get("candidate_count") != 500
            or screened.get("lexical_near_duplicate_count")
            != anchor["lexical_near_duplicate_count"]
            or screened.get("algorithm") != "casefolded_unicode_word_5gram_jaccard_v1"
            or screened.get("interpretation")
            != "automatic lexical screening only; semantic and human review pending"
        ):
            raise ValueError("Lexical near-duplicate screen differs from frozen CPU audit")
    if verify_aggregate:
        manifest = report.get("manifest_sha256")
        mutable = {k: v for k, v in report.items() if k != "manifest_sha256"}
        computed = hashlib.sha256(canonical_json(mutable)).hexdigest()
        if computed != manifest or manifest != anchor.get("audit_manifest_sha256"):
            raise ValueError("Overall frozen 500-row metadata fingerprint mismatch")
    return {
        "status": "frozen_candidates_verified_unreviewed",
        "sources": 10,
        "groups": 500,
        "discovery": 200,
        "evaluation": 300,
        "audit_sha256": anchor["audit_manifest_sha256"],
        "public_source_text": False,
        "license_and_human_review": "pending",
        "gpu_measurements": 0,
        "rss_proposals_unreviewed": anchor.get("rss_variant_proposals_target", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Redacted JSON audit manifest")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args(argv)
    report = read_json(args.audit)
    lock = read_json(args.lock)
    print(json.dumps(verify_corpus_lock(report, lock), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
