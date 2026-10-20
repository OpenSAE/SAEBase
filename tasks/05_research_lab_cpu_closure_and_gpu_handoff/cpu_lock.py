"""Section 3 CPU source-lock replay. Raw text stays in the private experiment tree."""

from __future__ import annotations

import sys
from pathlib import Path

from common import LAB, REPO, file_sha256, log, write_json

sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO))

from backend.acquire import acquire  # noqa: E402
from backend.corpus_lock import verify_corpus_lock  # noqa: E402
from backend.core import read_json  # noqa: E402
from backend.ingest import _raw_jsonl, freeze_dataset  # noqa: E402
from backend.source_audit import main as audit_main  # noqa: E402


LOCK = LAB / "data" / "source_lock_v1.json"


def _summary(report: dict, verified: dict) -> dict:
    variants = report["rss_variant_proposals"]
    discovery = sum(row.get("discovery", 0) for row in report["dataset_results"])
    evaluation = sum(row.get("evaluation", 0) for row in report["dataset_results"])
    return {
        "sources": report["sources_selected"],
        "groups": report["candidate_count"],
        "discovery": discovery,
        "evaluation": evaluation,
        "variant_proposals": variants["proposed_variants"],
        "selection_proposals": variants["selection_proposals"],
        "held_out_proposals": variants["held_out_proposals"],
        "variant_metadata_sha256": variants["metadata_only_sha256"],
        "gpu_measurements": report["gpu_calls"],
        "cross_source_exact_duplicates": len(report["cross_source_exact_duplicates"]),
        "lexical_near_duplicate_count": report["lexical_near_duplicate_screen"].get(
            "lexical_near_duplicate_count"
        ),
        "human_reviewed": False,
        "license_reviewed": False,
        "full_source_text_in_audit": report["full_source_text_in_this_artifact"],
        "lock_verification": verified,
        "passed": (
            report["sources_selected"] == 10
            and report["candidate_count"] == 500
            and discovery == 200
            and evaluation == 300
            and variants["proposed_variants"] == 22500
            and variants["selection_proposals"] == 7500
            and variants["held_out_proposals"] == 15000
            and report["gpu_calls"] == 0
            and report["full_source_text_in_this_artifact"] is False
            and report["cross_source_exact_duplicates"] == []
        ),
    }


def run_cpu_lock(root: Path) -> dict:
    import datasets

    major = int(datasets.__version__.split(".", 1)[0])
    if major < 3 or major >= 5:
        raise RuntimeError(
            "Work Plan 05 source lock replays only with the CI pin datasets>=3,<5; "
            f"found datasets {datasets.__version__}, whose streaming shuffle does not preserve export hashes"
        )
    private = root / "private"
    audit_path = private / "opensae-corpus-audit.json"
    log(root, "CPU source audit starting")
    if not audit_path.exists():
        code = audit_main(
            [
                "--source-lock",
                str(LOCK),
                "--max-raw-rows",
                "2500",
                "--output",
                str(audit_path),
            ]
        )
        if code != 0:
            raise RuntimeError(
                "source_audit failed; stopping because the candidate lock may have changed"
            )
    report = read_json(audit_path)
    verified = verify_corpus_lock(report, read_json(LOCK))
    summary = _summary(report, verified)
    if not summary["passed"]:
        write_json(root / "GPU-00" / "cpu_lock_summary.json", summary)
        raise RuntimeError("CPU lock counts do not match Work Plan 05; refusing to continue")
    lock = read_json(LOCK)
    corpus = private / "corpus"
    for entry in lock["sources"]:
        dataset_id = entry["dataset_id"]
        frozen = corpus / dataset_id
        if (frozen / "manifest.json").exists():
            meta = read_json(frozen / "manifest.json")
        else:
            export = private / "exports" / f"{dataset_id}.jsonl"
            exported = acquire(
                dataset_id,
                export,
                max_raw_rows=int(entry["max_raw_rows"]),
                source_revision=entry["source_revision"],
            )
            if exported["source_export_sha256"] != entry["source_export_sha256"]:
                raise RuntimeError("Source export hash drifted for " + dataset_id)
            meta = freeze_dataset(
                dataset_id,
                _raw_jsonl(export),
                source_revision=exported["resolved_revision"],
                output=frozen,
                license_reviewed=False,
                human_reviewed=False,
            )
        if meta["records_sha256"] != entry["candidate_sha256"]:
            raise RuntimeError(
                "Private candidate hash does not match the source lock for " + dataset_id
            )
        if meta["human_reviewed"] or meta["license_reviewed"]:
            raise RuntimeError("Refusing forged review flags for " + dataset_id)
    summary["private_corpus"] = str(corpus)
    summary["audit_sha256"] = file_sha256(audit_path)
    summary["lock_sha256"] = file_sha256(LOCK)
    write_json(root / "GPU-00" / "cpu_lock_summary.json", summary)
    log(root, "CPU source lock matched 10/500/200/300/22500")
    return summary
