"""Task output evidence checks and status classification (offline)."""

from __future__ import annotations

import pytest

from tasks.precheck.worker import _validate_task, check


def test_task_cannot_claim_pass_from_configured_only():
    with pytest.raises(RuntimeError, match="did not return an executed"):
        _validate_task("capture", {"status": "configured_only", "operation": "capture"})


def test_task_requires_real_typed_artifacts():
    with pytest.raises(RuntimeError, match="features"):
        _validate_task(
            "capture", {"status": "executed", "operation": "capture", "feature_count": 0}
        )
    with pytest.raises(RuntimeError, match="metric"):
        _validate_task("benchmark", {"status": "executed", "operation": "benchmark", "metrics": []})


def test_worker_errors_do_not_claim_real_pass():
    result = check({"kind": "unknown"})
    assert result["status"] == "FAIL"
    assert result["code"] == "EXECUTION_ERROR"


def test_native_circuits_and_train_require_semantic_validation(tmp_path, monkeypatch):
    from tasks.precheck.report import error_result

    monkeypatch.chdir(tmp_path)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "native_graph.pt").write_bytes(b"native")
    with pytest.raises(RuntimeError, match="BLOCKED_VALIDATION") as circuit_error:
        _validate_task(
            "circuits",
            {
                "status": "executed",
                "operation": "circuits",
                "native": {"native_json_requires_source_specific_converter": True},
            },
        )
    outcome = error_result(circuit_error.value, stage="circuits", attempted=["real graph"])
    assert outcome["status"] == "BLOCKED"
    assert outcome["code"] == "BLOCKED_VALIDATION"
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "sae.safetensors").write_bytes(b"native")
    with pytest.raises(RuntimeError, match="reload"):
        _validate_task(
            "train",
            {
                "status": "executed",
                "operation": "train",
                "native": {"checkpoints_need_registration": True},
            },
        )
