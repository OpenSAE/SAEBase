"""CPU-only audit of the ten upstream source datasets.

Collect and freeze an internal unreviewed 50-prompt candidate batch on an
Internet-connected CPU runner. Only redacted hashes, source identities and
validation findings are exported to a shareable summary. No original source
text or model data is uploaded automatically.

The audit is *not* a human license review, a scientifically representative
sampling proof, or permission to publish third-party text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .acquire import acquire
from .core import canonical_json, dataset_catalog
from .ingest import _atomic_bytes, _raw_jsonl, freeze_dataset
from .near_duplicates import scan_near_duplicates
from .variant_audit import summarize_variant_proposals

SCHEMA = "opensae.corpus_cpu_audit.v1"


def _redacted_records(path: Path) -> list[dict[str, Any]]:
    output = []
    for record in _raw_jsonl(path):
        output.append(
            {
                "dataset_id": record["dataset_id"],
                "source_row_id": record["source_row_id"],
                "source_group_id": record["source_group_id"],
                "sample_id": record["sample_id"],
                "prompt_group_id": record["prompt_group_id"],
                "usage_split": record["usage_split"],
                "source_revision": record["source_revision"],
                "render_protocol": record["render_protocol"],
                "text_sha256": hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
                "no_republish_before_review": True,
            }
        )
    return output


def audit_source(
    dataset_id: str,
    *,
    root: Path,
    max_raw_rows: int = 2500,
    pinned_revision: str | None = None,
) -> dict[str, Any]:
    """Never returns prompt text. All raw files stay in caller's scratch dir."""
    root.mkdir(parents=True, exist_ok=True)
    source_file = root / "raw" / (dataset_id + ".jsonl")
    frozen = root / "frozen" / dataset_id
    source_file.parent.mkdir(parents=True, exist_ok=True)
    exported = acquire(
        dataset_id,
        source_file,
        max_raw_rows=(
            35000
            if dataset_id == "squad"
            else 20000
            if dataset_id == "wikitext103"
            else max_raw_rows
        ),
        source_revision=pinned_revision,
    )
    release = freeze_dataset(
        dataset_id,
        _raw_jsonl(source_file),
        source_revision=exported["resolved_revision"],
        output=frozen,
        license_reviewed=False,
        human_reviewed=False,
    )
    reviewed = _redacted_records(frozen / "records.jsonl")
    if len(reviewed) != 50 or len({x["prompt_group_id"] for x in reviewed}) != 50:
        raise ValueError("Insufficient distinct input groups after grouping")
    if len({x["text_sha256"] for x in reviewed}) != 50:
        raise ValueError("Duplicate rendered texts within one source")
    return {
        "status": "candidate_frozen_unreviewed",
        "dataset_id": dataset_id,
        "source_repo": exported["repo"],
        "source_config": exported["config"],
        "source_split": exported["split"],
        "source_revision": exported["resolved_revision"],
        "source_export_sha256": exported["source_export_sha256"],
        "candidate_sha256": release["records_sha256"],
        "selection_seed": release["seed"],
        "selection_scope": exported["sampling_scope"],
        "max_raw_rows": exported["max_raw_rows"],
        "n_candidates": 50,
        "discovery": 20,
        "evaluation": 30,
        "license_reviewed": False,
        "human_reviewed": False,
        "user_can_reconstruct_from_pinned_source": True,
        "candidate_rows_without_text": reviewed,
    }


def audit_sources(
    *,
    destination: Path,
    dataset_ids: list[str] | None = None,
    max_raw_rows: int = 2500,
    revisions: dict[str, str] | None = None,
) -> dict[str, Any]:
    catalog = dataset_catalog()["sources"]
    all_ids = [item["id"] for item in catalog]
    chosen = all_ids if dataset_ids is None else dataset_ids
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(all_ids):
        raise ValueError("Unknown or repeated data source requested")
    if not 100 <= max_raw_rows <= 200000:
        raise ValueError("CPU dataset scan budget must be between 100 and 200000")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    pinned = revisions or {}
    if not set(pinned) <= set(chosen):
        raise ValueError("Unexpected pinned source revision")
    if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in pinned.values()):
        raise ValueError("Pinned source commits must be full 40-character Git hashes")
    results = []
    with tempfile.TemporaryDirectory(prefix="opensae-corpus-unreviewed-") as temp:
        scratch = Path(temp)
        for name in chosen:
            try:
                results.append(
                    audit_source(
                        name,
                        root=scratch,
                        max_raw_rows=max_raw_rows,
                        pinned_revision=pinned.get(name),
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "dataset_id": name,
                        "status": "blocked",
                        "error_class": type(exc).__name__,
                        "error_summary": str(exc).splitlines()[0][:280],
                        "candidate_rows_without_text": [],
                    }
                )
        lexical = scan_near_duplicates(scratch / "frozen", chosen)
        variants = summarize_variant_proposals(scratch / "frozen", chosen)
    complete = [row for row in results if row["status"] == "candidate_frozen_unreviewed"]
    all_rows = [row for entry in complete for row in entry["candidate_rows_without_text"]]
    seen: dict[str, dict[str, Any]] = {}
    conflicts = []
    for record in all_rows:
        fingerprint = record["text_sha256"]
        if fingerprint in seen:
            conflicts.append(
                {
                    "left": seen[fingerprint]["sample_id"],
                    "right": record["sample_id"],
                    "cross_usage_split": seen[fingerprint]["usage_split"] != record["usage_split"],
                }
            )
        else:
            seen[fingerprint] = record
    result = {
        "schema": SCHEMA,
        "status": "candidate_metadata_only_not_licensed_or_human_reviewed",
        "number_sources_target": len(chosen),
        "sources_selected": len(complete),
        "base_prompt_target": len(chosen) * 50,
        "candidate_count": len(all_rows),
        "cross_source_exact_duplicates": conflicts,
        "cross_source_near_duplicate_review": "automatic_fivegram_screen_only_human_review_pending",
        "lexical_near_duplicate_screen": lexical,
        "rss_variant_proposals": variants,
        "human_content_review": "not_completed",
        "rights_review": "not_completed",
        "dataset_results": results,
        "gpu_calls": 0,
        "full_source_text_in_this_artifact": False,
    }
    # The digest includes only redacted source row identifiers and hashes.
    result["manifest_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    _atomic_bytes(destination, canonical_json(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", help="Repeat to scan selected sources")
    parser.add_argument("--max-raw-rows", type=int, default=2500)
    parser.add_argument("--revisions", type=Path, help="Optional JSON dataset_id -> 40-hex SHA")
    parser.add_argument(
        "--source-lock", type=Path, help="Use frozen source commits from source_lock_v1.json"
    )
    args = parser.parse_args(argv)
    if args.revisions and args.source_lock:
        parser.error("Use either --source-lock or --revisions, not both")
    if args.source_lock:
        locked = json.loads(args.source_lock.read_text(encoding="utf-8"))
        if locked.get("schema") != "opensae.corpus_source_lock.v1":
            parser.error("Invalid pinned corpus source lock schema")
        versions = {row["dataset_id"]: row["source_revision"] for row in locked["sources"]}
    else:
        versions = json.loads(args.revisions.read_text(encoding="utf-8")) if args.revisions else {}
    report = audit_sources(
        destination=args.output,
        dataset_ids=args.source,
        max_raw_rows=args.max_raw_rows,
        revisions=versions,
    )
    summary = {
        "sources_selected": report["sources_selected"],
        "candidate_count": report["candidate_count"],
        "failures": [
            {"source": row["dataset_id"], "error": row.get("error_summary", "")}
            for row in report["dataset_results"]
            if row["status"] == "blocked"
        ],
        "reviewed_for_public_release": False,
        "metadata_file": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if not summary["failures"] and not report["cross_source_exact_duplicates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
