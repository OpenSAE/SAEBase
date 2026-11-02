"""Real byte-hash and receipt-negative coverage; no GPU or model download."""
import hashlib
import json

from saebase.runtime.audit_report import audit_run
from saebase.runtime.audit_delivery import deliver


def _run(tmp_path, status="BLOCKED"):
    root = tmp_path / "run"
    root.mkdir()
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({"status": status, "run_id": "r1", "source_sha": "abc"}))
    data = receipt.read_bytes()
    manifest = {
        "schema": "saebase.artifact_manifest.v1",
        "run_id": "r1",
        "source_sha": "abc",
        "files": [{"path": "receipt.json", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
        "checks": [{"id": "precheck:model1", "kind": "precheck", "receipt": "receipt.json"}],
    }
    return root, receipt, manifest


def test_blocked_native_run_can_pass_evidence_audit(tmp_path):
    root, _, manifest = _run(tmp_path)
    r = audit_run(root, manifest)
    assert r["evidence_status"] == "PASS"
    assert r["execution_status"] == "BLOCKED"
    assert r["scientific_acceptance"] == "NOT_COVERED"


def test_missing_requested_receipt_blocks_delivery(tmp_path):
    root, _, manifest = _run(tmp_path)
    manifest["checks"].append({"id": "task:missing", "kind": "task", "receipt": "missing.json"})
    r = audit_run(root, manifest)
    assert r["evidence_status"] == "BLOCKED"
    assert deliver(root, manifest, tmp_path / "archives")["archive_status"] == "BLOCKED"


def test_tampered_file_blocks_archive(tmp_path):
    root, receipt, manifest = _run(tmp_path)
    receipt.write_text('{"status":"PASS"}')
    assert audit_run(root, manifest)["evidence_status"] == "BLOCKED"
    assert deliver(root, manifest, tmp_path / "archives")["archive_status"] == "BLOCKED"


def test_archive_true_default_and_immutable(tmp_path):
    root, _, manifest = _run(tmp_path, status="FAIL")
    dest = tmp_path / "archives"
    first = deliver(root, manifest, dest)
    assert first["archive_status"] == "ARCHIVED"
    assert first["remote_backup"] == "NOT_AUTHORIZED"
    assert first["sha256"]
    second = deliver(root, manifest, dest)
    assert second["archive_status"] == "BLOCKED"


def test_archive_false_still_generates_skip_receipt(tmp_path):
    root, _, manifest = _run(tmp_path)
    receipt = deliver(root, manifest, tmp_path / "archives", archive=False)
    assert receipt["archive_status"] == "SKIPPED"
    assert not (tmp_path / "archives").exists()


def test_wrong_source_identity_blocks(tmp_path):
    root, receipt, manifest = _run(tmp_path)
    receipt.write_text(json.dumps({"status": "PASS", "run_id": "r1", "source_sha": "different"}))
    data = receipt.read_bytes()
    manifest["files"][0].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    assert audit_run(root, manifest)["evidence_status"] == "BLOCKED"
