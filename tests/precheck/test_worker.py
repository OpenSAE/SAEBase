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
