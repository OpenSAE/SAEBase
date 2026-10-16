"""Pure-CPU verification and tamper detection for Stage D archives."""

import hashlib
import io
import tarfile

import pytest

from scripts.stage_d_archive import (
    ROOT, _relative_file, build_compact, digest, verify_bundle,
    verify_checksums, verify_compact, verify_original_archives,
    build_raw, verify_raw_bundle,
)


def test_committed_compact_evidence_has_complete_sha256_coverage():
    report = verify_compact(ROOT)
    assert report["checksummed_evidence_files"] == 199
    assert report["sites"] == 48
    assert report["population_benchmark"] == 48
    assert report["population_reliability"] == 48
    assert report["raw_original_archives_included"] is False


def test_compact_bundle_is_self_contained_and_can_be_independently_verified(tmp_path):
    archive = build_compact(ROOT, tmp_path, commit="a" * 40)
    assert archive.exists()
    assert (tmp_path / "SHA256SUMS").is_file()
    assert verify_bundle(archive)["sites"] == 48


def test_checksum_gate_detects_corruption_and_unlisted_files(tmp_path):
    base = tmp_path / "conf/coverage/stage_d_evidence"
    (base / "final").mkdir(parents=True)
    (base / "one.json").write_bytes(b'{"test":1}\n')
    sha = digest(base / "one.json")
    (base / "final/checksums.sha256").write_text(
        f"{sha}  one.json\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="199"):
        verify_checksums(tmp_path)
    (base / "one.json").write_bytes(b'{"test":2}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(tmp_path)


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsafe relative path"):
        _relative_file(tmp_path, "../outside")


def test_original_raw_hashes_and_repackaging(tmp_path):
    raw = tmp_path / "originals"
    raw.mkdir()
    file = raw / "one.tar.gz"
    with tarfile.open(file, "w:gz") as archive:
        contents = b'{"metrics":[]}\n'
        tarinfo = tarfile.TarInfo("runs/123/metrics.jsonl")
        tarinfo.size = len(contents)
        archive.addfile(tarinfo, io.BytesIO(contents))
    item = {"file": file.name, "bytes": file.stat().st_size, "sha256": digest(file)}
    index = {"archives": [item]}
    assert verify_original_archives(raw, index)["verified_archives"] == 1
    item["sha256"] = hashlib.sha256(b"fake").hexdigest()
    with pytest.raises(ValueError, match="digest/size"):
        verify_original_archives(raw, index)


def test_raw_archives_not_reconstructed_from_compact_evidence(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        verify_original_archives(tmp_path / "missing", {"archives": [{"file": "a.tar.gz"}]})
