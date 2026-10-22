#!/usr/bin/env python3
"""Prepare a SMALL WP13 handoff or a locally approved real Research Lab bundle.

By default this only indexes source-bound private trace metadata. It never
publishes prompt texts or activation arrays. An optional explicit reviewed
release builds actual SQLite/trace artifacts IN PLACE on the source host.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "apps" / "research-lab"))

from backend import explore  # noqa: E402
from backend.feature_store import create_feature_store, feature_store_count  # noqa: E402


MAX_FILES = 32
MAX_JSON_BYTES = 8 * 1024 * 1024
REVIEW_SCHEMA = "opensae.wp13.reviewed_traces.v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_traces(folder: Path) -> list[tuple[Path, str, dict[str, Any]]]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("A real private source directory is required")
    found: list[tuple[Path, str, dict[str, Any]]] = []
    for path in sorted(folder.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("No symlinked source files are accepted")
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Oversized raw trace; refuse silent truncation")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not {
            "site_id",
            "sample_id",
            "tokens",
            "model_sha256",
            "sae_sha256",
        }.issubset(record):
            continue
        digest = _sha(path)
        # Validate all scientific identity/token/code claims locally WITHOUT
        # granting actual publication approval or mutating raw flags.
        trial = {**record, "public_text_reviewed": True, "rights_reviewed": True}
        explore._valid_trace(trial, explore._sites())
        found.append((path, digest, record))
        if len(found) > MAX_FILES:
            raise ValueError("WP13 demo exceeds the current 32-trace viewer limit")
    if not found:
        raise ValueError("No genuine complete source-bound token traces found")
    ids = [(r["site_id"], r["sample_id"]) for _, _, r in found]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate original site/sample identities")
    return found


def _metadata(rows: list[tuple[Path, str, dict[str, Any]]]) -> dict[str, Any]:
    sites = Counter(r["site_id"] for _, _, r in rows)
    splits = Counter(r["split"] for _, _, r in rows)
    return {
        "schema": "opensae.wp13.small_gpu_handoff.v1",
        "source": "real_local_sae_traces_not_independently_certified",
        "public_release": False,
        "trace_count": len(rows),
        "token_count": sum(len(r["tokens"]) for _, _, r in rows),
        "observed_nonzero_codes": sum(
            len(token["features"]) for _, _, r in rows for token in r["tokens"]
        ),
        "sites": dict(sorted(sites.items())),
        "split_counts": dict(sorted(splits.items())),
        "reviewed_excerpt_count": sum(
            int(r.get("public_text_reviewed") is True) for _, _, r in rows
        ),
        "traces": [
            {
                "sample_id": record["sample_id"],
                "site_id": record["site_id"],
                "prompt_group_id": record["prompt_group_id"],
                "prompt_sha256": record["prompt_sha256"],
                "private_file_sha256": sha,
                "split": record["split"],
                "token_count": len(record["tokens"]),
                "nonzero_codes": sum(len(t["features"]) for t in record["tokens"]),
            }
            for _, sha, record in rows
        ],
        "privacy": "No prompt text, raw codes or model weights included.",
    }


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


def _observations(traces: list[dict[str, Any]]):
    for record in traces:
        for token in record["tokens"]:
            for code in token["features"]:
                yield {
                    "site_id": record["site_id"],
                    "model_sha256": record["model_sha256"],
                    "sae_sha256": record["sae_sha256"],
                    "source_type": "real_checkpoint",
                    "feature_id": code["feature_id"],
                    "token_position": token["position"],
                    "activation": code["activation"],
                    "prompt_group_id": record["prompt_group_id"],
                    "sample_id": record["sample_id"],
                    "token_text": token["text"],
                    "text_excerpt": record["prompt_text"][:700],
                    "public_text_reviewed": True,
                }


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


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
