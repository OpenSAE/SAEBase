"""Core functionality and independent release are distinct pass/fail axes."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from saebase.release.capability_acceptance import audit_ledger

ROOT = Path(__file__).resolve().parents[2]


def test_core_pass_matrix_is_specific_and_not_release_certification():
    file = ROOT / "harness" / "github_work_plan" / "12_capability_acceptance_matrix.csv"
    report = audit_ledger(file)
    assert report["capabilities"] == 50
    assert report["status_counts"]["PASS"] == 48
    assert report["status_counts"]["PARTIAL"] == 0
    assert report["status_counts"]["BLOCKED"] == 2
    assert report["p8_pending_count"] == 50
    assert report["release_ready"] is False
    with file.open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    assert rows["07"]["current_status"] == "PASS"
    assert rows["08"]["current_status"] == "PASS"
    assert rows["30"]["current_status"] == "PASS"
    assert "circuit_intervene.py" in rows["30"]["core_test"]
    assert rows["43"]["current_status"] == "PASS"
    assert rows["43"]["core_grade"] == "3/3"
    assert rows["44"]["current_status"] == "PASS"
    assert "clt_intervene.py" in rows["44"]["core_test"]
    for feature in ("03", "10", "19", "44"):
        assert rows[feature]["current_status"] == "PASS"
        assert rows[feature]["core_grade"] == "3/3"
        assert rows[feature]["p8_status"] == "PENDING"
    assert rows["20"]["current_status"] == "BLOCKED"
    assert rows["50"]["current_status"] == "BLOCKED"
    assert not any(row["p8_status"] == "VERIFIED" for row in rows.values())


def test_invalid_core_pass_does_not_escape_schema_validation(tmp_path):
    source = ROOT / "harness" / "github_work_plan" / "12_capability_acceptance_matrix.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
    row = next(r for r in rows if r["id"] == "08")
    row["core_grade"] = "1/3"
    changed = tmp_path / "bad_ledger.csv"
    with changed.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="core PASS"):
        audit_ledger(changed)
