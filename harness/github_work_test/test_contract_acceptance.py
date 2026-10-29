"""Independent deterministic coverage of selection and acceptance gates."""
import json
import pytest

from harness.contracts import acceptance, registry
from harness.contracts.selection import audit_selection


def test_selection_detects_omitted_real_io_contracts():
    assert audit_selection(["CT-BASE", "CT-STRUCT"], ["tree_nodes", "bilingual_fields"]) == ["CT-LOCALE"]
    assert audit_selection(["CT-PRIVACY"], ["sensitive_output"]) == []
    with pytest.raises(registry.ContractError):
        audit_selection([], ["unexpected_semantic_guess"])


def test_independent_gate_never_approves_and_rejects_missing_checks(tmp_path, monkeypatch):
    runs = tmp_path / "harness/tests/wp18/runs/demo/H04"
    protected = tmp_path / "harness/tests/wp18/acceptance/demo/H04"
    runs.mkdir(parents=True)
    protected.mkdir(parents=True)
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    monkeypatch.setattr(acceptance, "ACCEPTANCE", tmp_path / "harness/tests/wp18/acceptance")
    monkeypatch.setattr(acceptance.trace, "check_receipt_file", lambda *a, **kw: {"contract_count": 8})
    monkeypatch.setattr(acceptance.registry, "resolve", lambda *a: {"contract_ids": ["CT-PRIVACY"]})
    receipt = runs / "contract_receipt.json"
    receipt.write_text("{}")
    evaluator = protected / "evaluation.json"
    data = {
        "schema": "saebase.harness.independent_evaluation.v1",
        "case_id": "H04",
        "task_signals": ["sensitive_output"],
        "source_verified": True,
        "tests": [{"name": "nested negative", "passed": True}],
        "artifacts": [{"kind": "structured_output", "payload": {"synthetic": True, "gpu": None}}],
    }
    evaluator.write_text(json.dumps(data))
    result = acceptance.review(plan="unused", case="H04", receipt_path=receipt, evaluator_path=evaluator)
    assert result["status"] == "TECHNICAL_CHECKS_PASSED_NEEDS_HUMAN_REVIEW"
    assert result["human_approval"] is False
    data["artifacts"][0]["payload"]["device"] = {"token": "SYNTHETIC-SECRET"}
    evaluator.write_text(json.dumps(data))
    assert acceptance.review(plan="unused", case="H04", receipt_path=receipt, evaluator_path=evaluator)["status"] == "FAIL"
    data["artifacts"] = []
    data["tests"] = [{"name": "negative", "passed": False}]
    evaluator.write_text(json.dumps(data))
    assert acceptance.review(plan="unused", case="H04", receipt_path=receipt, evaluator_path=evaluator)["status"] == "FAIL"
    data["tests"] = [{"name": "negative", "passed": True}]
    data["task_signals"] = ["bilingual_fields"]
    evaluator.write_text(json.dumps(data))
    assert "SELECTION_GAP" in str(acceptance.review(plan="unused", case="H04", receipt_path=receipt, evaluator_path=evaluator)["issues"])
    with pytest.raises(registry.ContractError):
        acceptance.review(plan="unused", case="H04", receipt_path=receipt, evaluator_path=receipt)
