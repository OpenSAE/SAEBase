"""Historical archive checks without rewriting or retesting frozen material."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.github import archive


def _put(root: Path, path: str, data: bytes) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return archive.git_blob(data)


def _index(rows):
    return {
        "schema": archive.SCHEMA, "entries": rows, "prepared_for_versions": [],
    }


def test_repository_archive_preserves_exact_historical_git_blobs():
    index = archive.read_index()
    result = archive.check(index)
    assert result["status"] == "PASS"
    assert result["records"] >= 10
    assert "0.6.5" in result["versions"]


def test_moved_wt_is_excluded_from_default_pytest_and_original_is_absent():
    root = archive.ROOT
    assert not (root / "harness/github/work_test/test_v050_beta_release.py").exists()
    test_path = (root / "harness/github/work_test/archive/0.5.0/"
                 "0.5.0_wt001_beta_release/test_v050_beta_release.py")
    assert test_path.is_file()
    pyproject = (root / "pyproject.toml").read_text()
    assert 'norecursedirs = ["archive"]' in pyproject


def test_copy_compat_requires_same_source_and_archive_bytes(tmp_path: Path):
    original = "harness/github/development_doc/old.md"
    archived = (
        "harness/github/development_doc/archive/0.2.0/"
        "0.2.0_dev001_release_notes.md"
    )
    blob = _put(tmp_path, original, b"unchanged source bytes\n")
    _put(tmp_path, archived, b"unchanged source bytes\n")
    row = dict(
        original_path=original, archived_path=archived, git_blob_sha=blob,
        mode="copy_compat", status="archived", version="0.2.0",
    )
    assert archive.check(_index([row]), tmp_path)["status"] == "PASS"
    (tmp_path / archived).write_bytes(b"tampered\n")
    with pytest.raises(archive.ArchiveError, match="Historical bytes changed"):
        archive.check(_index([row]), tmp_path)


def test_move_requires_old_path_removed(tmp_path: Path):
    old = "harness/github/work_plan/report.md"
    target = (
        "harness/github/work_plan/archive/0.6.5/"
        "0.6.5_wp001_exam_closeout.md"
    )
    blob = _put(tmp_path, target, b"frozen\n")
    row = dict(original_path=old, archived_path=target, git_blob_sha=blob,
               mode="move", status="archived", version="0.6.5")
    assert archive.check(_index([row]), tmp_path)["status"] == "PASS"
    _put(tmp_path, old, b"frozen\n")
    with pytest.raises(archive.ArchiveError, match="still exists"):
        archive.check(_index([row]), tmp_path)


@pytest.mark.parametrize("path", [
    "harness/github/work_plan/archive/0.6.5/0.6.5_wp1_long_bad.md",
    "harness/github/development_doc/archive/0.10.0/0.9.0_dev001_qa_smoke.md",
    "harness/github/work_test/archive/0.5.0/0.5.0_wt001_wrong/test.py",
    "/tmp/private", "harness/github/work_plan/../doc.md",
])
def test_noncanonical_names_rejected(path):
    with pytest.raises(archive.ArchiveError):
        archive.archive_identity(path)


def test_older_release_still_valid_but_newer_requires_preparation(tmp_path: Path):
    target = (
        "harness/github/work_plan/archive/0.6.5/"
        "0.6.5_wp001_exam_closeout.md"
    )
    blob = _put(tmp_path, target, b"frozen\n")
    row = dict(
        original_path="harness/github/work_plan/old.md",
        archived_path=target, git_blob_sha=blob, mode="move",
        status="archived", version="0.6.5",
    )
    index = _index([row])
    archive.check(index, tmp_path, release_version="0.6.5")
    with pytest.raises(archive.ArchiveError, match="requires archive preparation"):
        archive.check(index, tmp_path, release_version="0.7.0")
    index["prepared_for_versions"] = ["0.7.0"]
    archive.check(index, tmp_path, release_version="0.7.0")


def test_archive_registry_serialization_and_sequential_numbering(tmp_path: Path):
    row = {
        "original_path": "harness/github/work_plan/old.md",
        "archived_path": "harness/github/work_plan/archive/0.6.5/"
        "0.6.5_wp001_qa_smoke.md",
        "git_blob_sha": _put(
            tmp_path,
            "harness/github/work_plan/archive/0.6.5/"
            "0.6.5_wp001_qa_smoke.md",
            b"source unchanged\n",
        ),
        "mode": "move", "status": "archived", "version": "0.6.5",
    }
    (tmp_path / archive.INDEX).parent.mkdir(parents=True)
    (tmp_path / archive.INDEX).write_text(json.dumps(_index([row])))
    assert archive.read_index(tmp_path)["entries"][0] == row
    with pytest.raises(archive.ArchiveError, match="Duplicate"):
        archive.check(_index([row, dict(row, archived_path=row["archived_path"])]), tmp_path)
