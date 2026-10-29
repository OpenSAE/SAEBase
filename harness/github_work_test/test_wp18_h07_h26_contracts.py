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


def test_verifier_executes_correct_and_incorrect_candidate(tmp_path, monkeypatch):
    sandbox = tmp_path / "test-run" / "H07"
    sandbox.mkdir(parents=True)
    monkeypatch.setattr(wp18_verify, "RUNS", tmp_path)
    source = sandbox / "implementation.py"
    source.write_text(
        "def solve(payload):\\n"
        "    paths = payload['paths']\\n"
        "    if not isinstance(paths, list) or len(paths) != len(set(paths)):\\n"
        "        raise ValueError('duplicate')\\n"
        "    if any(not isinstance(x, str) or not x or x.startswith('/')\\n"
        "           or any(p in ('', '.', '..') for p in x.split('/'))\\n"
        "           for x in paths):\\n"
        "        raise ValueError('bad path')\\n"
        "    return {'paths': sorted(paths)}\\n",
        encoding="utf-8",
    )
    correct = wp18_verify.run_case("H07", sandbox)
    assert correct["failed"] == 0
    assert correct["passed"] >= 4
    source.write_text("def solve(payload):\\n    return {'paths': []}\\n", encoding="utf-8")
    faulty = wp18_verify.run_case("H07", sandbox)
    assert faulty["failed"] >= 2
    assert faulty["human_approved"] is False


def test_scope_audit_rejects_governance_and_old_run_changes(monkeypatch):
    import types

    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(
            stdout="A\\tharness/tests/wp18/runs/new-run/H07/implementation.py\\n"
            "M\\tharness/contracts/catalog.json\\n"
        )

    monkeypatch.setattr(wp18_verify.subprocess, "run", fake_run)
    rejected = wp18_verify.audit_changed_scope("a" * 40)
    assert rejected == ["harness/contracts/catalog.json"]
