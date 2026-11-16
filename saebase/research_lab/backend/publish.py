"""Prepare approved local Lab bundles and publish immutable report summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import explore
from .core import canonical_json, data_root
from .feature_store import (
    _metadata,
    _observations,
    _read_traces,
    create_feature_store,
    feature_store_count,
)
from .ingest import _atomic_bytes


def publish_report(source: dict[str, Any], *, output: Path, public_consent: bool) -> dict[str, Any]:
    if not public_consent:
        raise PermissionError("Explicit public-text/result publication approval required")
    if source.get("visibility") != "public":
        raise ValueError("Source must explicitly request public visibility")
    if source.get("source_type") not in ("real_checkpoint", "synthetic_demo", "external_verified"):
        raise ValueError("Source type must be explicit")
    if source.get("kind") not in ("intervention", "sar", "rss", "localization", "benchmark"):
        raise ValueError("Unsupported report kind")
    if not isinstance(source.get("protocol_id"), str) or not source["protocol_id"]:
        raise ValueError("Missing protocol identity")
    if not isinstance(source.get("source"), dict) or not source["source"].get("commit"):
        raise ValueError("Missing source commit provenance")
    if not isinstance(source.get("title"), str) or not source["title"].strip():
        raise ValueError("Missing report title")
    if not isinstance(source.get("results"), list):
        raise ValueError("Missing measured results")
    cleaned = dict(source)
    cleaned.pop("report_id", None)
    for result in cleaned["results"]:
        if not isinstance(result, dict) or not {"metric", "status"} <= result.keys():
            raise ValueError("Malformed result row")
        if result["status"] not in ("completed", "blocked", "unsupported", "failed", "undefined"):
            raise ValueError("Unsupported result status")
        if result["status"] != "completed" and result.get("value") is not None:
            raise ValueError("Uncomputed result cannot have a numerical value")
    report_id = hashlib.sha256(canonical_json(cleaned)).hexdigest()[:40]
    cleaned["report_id"] = report_id
    folder = output / report_id
    if folder.exists():
        raise FileExistsError("This immutable report already exists")
    payload = canonical_json(cleaned)
    manifest = {
        "schema": "opensae.public_report.v1",
        "report_id": report_id,
        "report_sha256": hashlib.sha256(payload).hexdigest(),
        "source_type": cleaned["source_type"],
        "public_consent": True,
    }
    _atomic_bytes(folder / "report.json", payload)
    _atomic_bytes(folder / "manifest.json", canonical_json(manifest))
    return {"report_id": report_id, "path": str(folder), "report_sha256": manifest["report_sha256"]}


def report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--public-consent", action="store_true")
    args = parser.parse_args(argv)
    with args.input.open("r", encoding="utf-8") as f:
        report = json.load(f)
    print(
        json.dumps(
            publish_report(
                report,
                output=args.output or data_root() / "reports",
                public_consent=args.public_consent,
            ),
            indent=2,
        )
    )
    return 0


REVIEW_SCHEMA = "opensae.wp13.reviewed_traces.v1"


def _approved(
    rows: list[tuple[Path, str, dict[str, Any]]], review: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("public_consent") is not True
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("approved"), list)
        or not review["approved"]
    ):
        raise PermissionError("Explicit human/operator review and public consent required")
    indexed = {(r["site_id"], r["sample_id"]): (sha, r) for _, sha, r in rows}
    accepted = []
    visited = set()
    for item in review["approved"]:
        if not isinstance(item, dict):
            raise PermissionError("Invalid source-specific review item")
        key = (item.get("site_id"), item.get("sample_id"))
        if key in visited or key not in indexed:
            raise PermissionError("Duplicate, unknown or changed reviewed trace")
        visited.add(key)
        sha, record = indexed[key]
        if (
            item.get("private_file_sha256") != sha
            or item.get("prompt_sha256") != record["prompt_sha256"]
            or item.get("rights_reviewed") is not True
            or item.get("public_text_reviewed") is not True
        ):
            raise PermissionError("Trace bytes/prompt changed or review flags not approved")
        accepted.append({**record, "public_text_reviewed": True, "rights_reviewed": True})
    return accepted


def make_review_template(rows: list[tuple[Path, str, dict[str, Any]]]) -> dict:
    """Only a source-digest checklist, not any kind of automatic approval."""
    return {
        "schema": REVIEW_SCHEMA,
        "reviewer": "",
        "public_consent": False,
        "approved": [
            {
                "site_id": r["site_id"],
                "sample_id": r["sample_id"],
                "prompt_sha256": r["prompt_sha256"],
                "private_file_sha256": sha,
                "rights_reviewed": False,
                "public_text_reviewed": False,
            }
            for _, sha, r in rows
        ],
        "warning": "Review actual prompt and permitted release scope locally before changing any flag. This does not authorize public publishing.",
    }


def bundle_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-raw", type=Path, required=True)
    p.add_argument("--small-report", type=Path, required=True)
    p.add_argument("--review-template", type=Path)
    p.add_argument("--review-file", type=Path)
    p.add_argument("--publish-reviewed", action="store_true")
    p.add_argument("--release-root", type=Path)
    args = p.parse_args(argv)

    rows = _read_traces(args.private_raw)
    summary = _metadata(rows)
    if args.small_report.exists():
        p.error("Refusing to overwrite a previous handoff report")
    args.small_report.parent.mkdir(parents=True, exist_ok=True)
    args.small_report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.review_template is not None:
        if args.review_template.exists():
            p.error("Refusing to overwrite a previous review template")
        args.review_template.parent.mkdir(parents=True, exist_ok=True)
        args.review_template.write_text(
            json.dumps(make_review_template(rows), indent=2, sort_keys=True) + "\n"
        )
    if args.publish_reviewed:
        if args.review_file is None or args.release_root is None:
            p.error("--publish-reviewed requires --review-file and --release-root")
        if args.release_root.exists():
            p.error("A reviewed release output must be new and immutable")
        review = json.loads(args.review_file.read_text(encoding="utf-8"))
        approved = _approved(rows, review)
        if not approved:
            p.error("No source/rights-approved trace was selected")
        args.release_root.mkdir(mode=0o700, parents=True)
        feature_meta = create_feature_store(_observations(approved), args.release_root / "features")
        trace_meta = explore.publish_exploration(
            approved, [], output=args.release_root / "explore", public_consent=True
        )
        manifest = {
            "schema": "opensae.wp13.approved_public_bundle.v1",
            "publication": "ready_for_staging_only_not_deployed",
            "trace_count": trace_meta["trace_count"],
            "feature_count": feature_meta["feature_count"],
            "sqlite_sha256": feature_meta["files"]["features.sqlite3"],
            "trace_sha256": trace_meta["files"]["records.json"],
            "rights_reviewed": True,
            "source": "real_local_sae_traces",
        }
        (args.release_root / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if feature_store_count(args.release_root) != manifest["feature_count"]:
            raise RuntimeError("Staging bundle cannot load its immutable SQLite index")
        if len(explore.published_exploration(args.release_root)["traces"]) != len(approved):
            raise RuntimeError("Staging bundle cannot open reviewed token traces")
        print(json.dumps({"built": True, "publicly_deployed": False, **manifest}))
    else:
        print(
            json.dumps(
                {
                    "built": False,
                    "publicly_deployed": False,
                    "count": summary["trace_count"],
                    "reviewed_excerpt_count": summary["reviewed_excerpt_count"],
                    "small_report": str(args.small_report),
                }
            )
        )
    return 0


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bundle":
        del sys.argv[1]
        return bundle_main()
    return report_main()


if __name__ == "__main__":
    raise SystemExit(main())
