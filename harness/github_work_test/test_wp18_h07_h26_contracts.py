"""Governance-owned proof that the twenty fixed verification contracts are active."""

from __future__ import annotations

import json

from harness.contracts import registry
from harness.contracts import wp18_verify
from harness.contracts.selection import audit_selection


def test_twenty_cases_are_frozen_with_positive_and_negative_checks():
    state = wp18_verify.validate_spec()
    assert state["cases"] == 20
    assert state["checks"] >= 80
    assert state["contracts"] >= 20


def test_every_case_is_bound_by_workplan_and_io_contracts():
    specs = wp18_verify.load_spec()["cases"]
    for item in specs:
        selection = registry.resolve(wp18_verify.PLAN, item["id"])
        assert selection["contract_ids"]
        assert audit_selection(selection["contract_ids"], item["signals"]) == []


def test_new_io_contracts_are_not_silently_dropped():
    with_status = registry.resolve(wp18_verify.PLAN, "H23")["contract_ids"]
    assert audit_selection(with_status, ["release_gate", "review_state"]) == []
    assert audit_selection(["CT-BASE"], ["release_gate"]) == ["CT-RELEASE"]


def test_wrong_oracle_never_self_certifies():
    cases = wp18_verify.load_spec()["cases"]
    for case in cases:
        assert any(row.get("reject") is True for row in case["checks"])
        assert any("expected" in row for row in case["checks"])
        assert len({row["name"] for row in case["checks"]}) == len(case["checks"])


def test_structured_spec_has_no_production_execution_steps():
    payload = json.dumps(wp18_verify.load_spec(), ensure_ascii=False)
    assert "ssh " not in payload
    assert "sudo " not in payload
    assert "torch.load(" not in payload
    assert "git push" not in payload
