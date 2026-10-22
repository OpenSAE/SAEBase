"""WP13 compact GPU handoff and operator-reviewed local release regression tests.

The fixtures are artificial, never presented as real user-facing SAE measurements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "research-lab"))

from backend import core, explore  # noqa: E402
from backend.feature_store import feature_store_count  # noqa: E402
from scripts.wp13_prepare_public_bundle import (  # noqa: E402
    _approved,
    _metadata,
    _read_traces,
    main,
    make_review_template,
)


def _trace(site: dict, index: int) -> dict:
    prompt = f"An original short engineering demonstration {index}."
    return {
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "source_type": "real_checkpoint",
        "sample_id": f"test-L5-{index}",
        "prompt_group_id": f"test-group-{index}",
        "split": "discovery" if index == 1 else "evaluation",
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tokenizer_sha256": "a" * 64,
        "model_processing": "synthetic-offline-test-only",
        "complete_sparse_codes": True,
        "public_text_reviewed": False,
        "rights_reviewed": False,
        "tokens": [
            {
                "position": 0, "token_id": 2, "text": "<bos>",
                "features": [{"feature_id": 923, "activation": 3405.25}],
            },
            {
                "position": 1, "token_id": 505, "text": "Paris",
                "features": [{"feature_id": 506, "activation": float(index) * 3}],
            },
        ],
    }


def _files(tmp_path):
    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    folder = tmp_path / "private-raw"
    folder.mkdir()
    for n in (1, 2):
        (folder / f"site-L5-{n}.json").write_text(
            json.dumps(_trace(site, n), ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return folder


def test_metadata_only_handoff_never_exposes_prompt_or_feature_values(tmp_path):
    source = _files(tmp_path)
    report = tmp_path / "summary.json"
    checklist = tmp_path / "review-template.json"
    assert main([
        "--private-raw", str(source),
        "--small-report", str(report),
        "--review-template", str(checklist),
    ]) == 0
    output = json.loads(report.read_text())
    assert output["trace_count"] == 2
    assert output["observed_nonzero_codes"] == 4
    assert output["reviewed_excerpt_count"] == 0
    assert len(output["traces"]) == 2
    assert "An original" not in report.read_text()
    assert "Paris" not in report.read_text()
    assert "3405.25" not in report.read_text()
    review = json.loads(checklist.read_text())
    assert review["public_consent"] is False
    assert all(item["rights_reviewed"] is False for item in review["approved"])
    assert not (tmp_path / "public").exists()


def test_private_source_remains_immutable_and_approval_is_hash_bound(tmp_path):
    source = _files(tmp_path)
    records = _read_traces(source)
    summary = _metadata(records)
    assert summary["sites"][records[0][2]["site_id"]] == 2
    raw_before = (source / "site-L5-1.json").read_bytes()
    template = make_review_template(records)
    with pytest.raises(PermissionError):
        _approved(records, template)
    template["reviewer"] = "Actual approved operator"
    template["public_consent"] = True
    template["approved"] = [template["approved"][0]]
    template["approved"][0]["rights_reviewed"] = True
    template["approved"][0]["public_text_reviewed"] = True
    accepted = _approved(records, template)
    assert len(accepted) == 1
    assert accepted[0]["public_text_reviewed"] is True
    assert records[0][2]["public_text_reviewed"] is False
    template["approved"][0]["private_file_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="changed"):
        _approved(records, template)
    assert (source / "site-L5-1.json").read_bytes() == raw_before


def test_approved_release_builds_actual_lab_index_and_explore_in_place(tmp_path):
    source = _files(tmp_path)
    rows = _read_traces(source)
    checklist = make_review_template(rows)
    checklist["reviewer"] = "Review completed for original owned texts"
    checklist["public_consent"] = True
    for row in checklist["approved"]:
        row["rights_reviewed"] = True
        row["public_text_reviewed"] = True
    approval = tmp_path / "review.json"
    approval.write_text(json.dumps(checklist), encoding="utf-8")
    bundle = tmp_path / "public-bundle"
    assert main([
        "--private-raw", str(source), "--small-report", str(tmp_path / "release-preflight.json"),
        "--review-file", str(approval), "--publish-reviewed",
        "--release-root", str(bundle),
    ]) == 0
    assert (bundle / "features" / "features.sqlite3").is_file()
    assert feature_store_count(bundle) == 2
    assert (bundle / "explore" / "manifest.json").is_file()
    released = explore.published_exploration(bundle)
    assert len(released["traces"]) == 2
    assert all(t["public_text_reviewed"] for t in released["traces"])
    assert all(t["rights_reviewed"] for t in released["traces"])
    assert (bundle / "release-manifest.json").is_file()
    with pytest.raises(SystemExit):
        main([
            "--private-raw", str(source), "--small-report", str(tmp_path / "other.json"),
            "--review-file", str(approval), "--publish-reviewed", "--release-root", str(bundle),
        ])
