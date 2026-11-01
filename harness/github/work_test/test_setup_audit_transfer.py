"""Study-independent Setup, Audit and B2 boundaries without GPU or network."""

from __future__ import annotations

import hashlib

import pytest

from saebase.setup.workflow import representative_sites, estimate_runtime
from saebase.runtime.audit import audit_artifacts
from harness.transfer.b2 import command, remote_path


def test_representatives_preserve_models_and_unverified_coverage():
    sites = [
        {"model_key": "small", "sae_key": "a", "layer": 3},
        {"model_key": "small", "sae_key": "a", "layer": 5},
        {"model_key": "large", "sae_key": "b", "layer": 8},
    ]
    choice = representative_sites(sites, 2)
    assert [x["model_key"] for x in choice["sites"]] == ["small", "large"]
    assert choice["unverified_sites"] == 1
    assert choice["coverage"] == "representative_only"
    assert representative_sites(sites, 1)["uncovered_models"] == ["large"]
    with pytest.raises(ValueError):
        representative_sites(sites + [sites[0]], 2)


def test_estimate_only_from_actual_pilot():
    assert estimate_runtime([], 100)["status"] == "NOT_RUN"
    report = estimate_runtime([{"samples": 10, "elapsed_seconds": 5}], 100)
    assert report["estimated_seconds"] == 50
    with pytest.raises(ValueError):
        estimate_runtime([{"samples": 10, "elapsed_seconds": 0}], 100)


def test_artifact_integrity_and_optional_block_threshold(tmp_path):
    f = tmp_path / "result.json"
    data = b'{"result":1}\n'
    f.write_bytes(data)
    good = {
        "path": "result.json",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "required": True,
    }
    manifest = {"schema": "saebase.artifact_manifest.v1", "files": [good]}
    assert audit_artifacts(tmp_path, manifest)["status"] == "PASS"
    f.write_bytes(b"altered")
    assert audit_artifacts(tmp_path, manifest)["status"] == "BLOCKED"
    f.write_bytes(data)
    missing = dict(good, path="optional.json", required=False)
    manifest["files"].append(missing)
    manifest["max_optional_failures"] = 1
    assert audit_artifacts(tmp_path, manifest)["status"] == "PARTIAL"
    manifest["max_optional_failures"] = 0
    assert audit_artifacts(tmp_path, manifest)["status"] == "BLOCKED"
    manifest["files"][1]["path"] = "../private.txt"
    with pytest.raises(ValueError, match="identity"):
        audit_artifacts(tmp_path, manifest)


def test_b2_preserves_dry_run_and_immutable_copy(tmp_path):
    argv = command("push", str(tmp_path), "b2:bucket/study/run_001")
    assert argv[:2] == ["rclone", "copy"]
    assert "--immutable" in argv and "--dry-run" in argv
    assert argv[4:6] == ["--transfers", "16"]
    assert "--fast-list" in argv
    assert "--b2-upload-concurrency" in argv
    assert "sync" not in argv and "purge" not in argv
    verify = command("verify", str(tmp_path), "b2:bucket/study/run_001")
    assert verify[1] == "check" and "--one-way" in verify
    with pytest.raises(ValueError):
        remote_path("b2:bucket")
    with pytest.raises(ValueError):
        command("push", str(tmp_path), "b2:bucket/study/run_001", transfers=0)
