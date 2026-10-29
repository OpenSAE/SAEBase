"""Independent checks for predeclared, reusable Harness contract obligations."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from harness.contracts import invariants, registry, trace

PLAN = "harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md"


def test_catalog_has_a_closed_and_versioned_vocabulary():
    catalog = registry.catalog()
    ids = {item["id"] for item in catalog["contracts"]}
    assert len(ids) >= 14
    assert set(catalog["required_always"]) == {
        "CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE"
    }
    assert all(item["requirements"] for item in catalog["contracts"])
    assert all(item["negative_examples"] for item in catalog["contracts"])


def test_wp18_all_cases_bind_predeclared_contracts():
    plan = registry.validate_plan(PLAN)
    assert set(plan["cases"]) == {"H01", "H02", "H03", "H04", "H05", "H06"}
    required = {
        "H01": "CT-EVIDENCE",
        "H02": "CT-STRUCT",
        "H03": "CT-CONTEXT",
        "H04": "CT-PRIVACY",
        "H05": "CT-REVIEW",
        "H06": "CT-LOCALE",
    }
    for case, specific in required.items():
        selection = registry.resolve(PLAN, case)
        assert set(registry.catalog()["required_always"]) <= set(selection["contract_ids"])
        assert specific in selection["contract_ids"]
        assert "CT-ERROR" in selection["contract_ids"]


def test_registry_rejects_missing_case_and_unknown_contract(monkeypatch):
    with pytest.raises(registry.ContractError):
        registry.resolve(PLAN, "H99")
    good = registry.declaration(PLAN)
    good["cases"]["H01"]["contracts"] = ["CT-FAKE"]
    monkeypatch.setattr(registry, "declaration", lambda _: good)
    with pytest.raises(registry.ContractError, match="Unknown contract"):
        registry.resolve(PLAN, "H01")


def test_registry_rejects_category_without_specific_contract(monkeypatch):
    declaration = registry.declaration(PLAN)
    declaration["cases"]["H03"]["contracts"] = ["CT-INTERFACE"]
    monkeypatch.setattr(registry, "declaration", lambda _: declaration)
    with pytest.raises(registry.ContractError, match="No specific contract"):
        registry.resolve(PLAN, "H03")


@pytest.mark.parametrize(
    "source",
    [
        "../private.md",
        "/etc/passwd",
        "harness/github_work_plan/../../private.md",
        "harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md/../x.md",
    ],
)
def test_registry_cannot_load_workplan_outside_source_root(source):
    with pytest.raises(registry.ContractError):
        registry.plan_path(source)


def test_custom_task_can_select_existing_contracts_without_inventing_new_id():
    selected = trace.ad_hoc_selection(["privacy", "interface"])
    assert set(registry.catalog()["required_always"]) <= set(selected)
    assert "CT-PRIVACY" in selected and "CT-INTERFACE" in selected
    with pytest.raises(registry.ContractError):
        trace.ad_hoc_selection(["new-unsafe-task-kind"])


def test_reference_validation_is_tiered_and_safe(tmp_path):
    f = tmp_path / "source.py"
    f.write_text("line1\nline2\n", encoding="utf-8")
    row = invariants.reference_location("source.py:2", tmp_path)
    assert row["syntax_valid"] and row["boundary_valid"] and row["source_resolved"]
    unchecked = invariants.reference_location("missing.py:5", tmp_path, resolve=False)
    assert unchecked["syntax_valid"] and not unchecked["source_resolved"]
    with pytest.raises(invariants.InvariantError):
        invariants.reference_location("missing.py:5", tmp_path, resolve=True)


@pytest.mark.parametrize(
    "reference",
    ("../secret.py:1", "/private/file.py:1", "x.py:0", "x.py:-2",
     "./x.py:1", "x//y.py:1", "x.py:not-a-line"),
)
def test_reference_does_not_confuse_path_syntax_with_verified_source(tmp_path, reference):
    with pytest.raises(invariants.InvariantError):
        invariants.reference_location(reference, tmp_path, resolve=False)


def test_evidence_symlink_escape_and_missing_line(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside.py")
    outside.write_text("token\n", encoding="utf-8")
    (tmp_path / "outside.py").symlink_to(outside)
    with pytest.raises(invariants.InvariantError):
        invariants.reference_location("outside.py:1", tmp_path)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(invariants.InvariantError):
        invariants.reference_location("a.txt:5", tmp_path)


def test_recursive_redaction_covers_dict_list_values_and_unknown_types():
    source = {"synthetic": True, "device": {"token": "SECRET-DO-NOT-LOG",
              "events": [{"private_key": "SUPERSECRET"},
                         {"home": "/Users/researcher/private"}]}, "gpu": None}
    cleaned, redactions = invariants.redact_structure(source)
    payload = json.dumps(cleaned)
    assert "SECRET-DO-NOT-LOG" not in payload
    assert "SUPERSECRET" not in payload
    assert "/Users/researcher" not in payload
    assert cleaned["synthetic"] is True and cleaned["gpu"] is None
    assert "device.token" in redactions
    assert "device.events[0].private_key" in redactions
    with pytest.raises(invariants.InvariantError):
        invariants.redact_structure({"payload": object()})


def test_redaction_never_silently_passes_unsupported_depth():
    with pytest.raises(invariants.InvariantError):
        invariants.redact_structure({"nested": {"secret": "x"}}, max_depth=0)


def test_tree_scope_rejects_absent_parent_without_fabricating_id():
    g = {"nodes": [{"id": "root", "parent_id": None, "tree_id": "code", "order": 0},
                   {"id": "child", "parent_id": "root", "tree_id": "code", "order": 2}]}
    assert invariants.validated_dfs(g) == ["root", "child"]
    assert invariants.validated_dfs(g, scope="parent", target="root") == ["root", "child"]
    assert invariants.validated_dfs(g, scope="node", target="child") == ["child"]
    with pytest.raises(invariants.InvariantError):
        invariants.validated_dfs(g, scope="parent", target="missing")
    with pytest.raises(invariants.InvariantError):
        invariants.validated_dfs(g, scope="tree", target="not-a-tree")


def test_tree_rejects_unknown_parent_cycle_and_cross_tree():
    with pytest.raises(invariants.InvariantError):
        invariants.validated_dfs({"nodes": [{"id": "a", "parent_id": "missing"}]})
    with pytest.raises(invariants.InvariantError):
        invariants.validated_dfs({"nodes": [
            {"id": "a", "parent_id": "b"}, {"id": "b", "parent_id": "a"}
        ]})
    with pytest.raises(invariants.InvariantError):
        invariants.validated_dfs({"nodes": [
            {"id": "a", "tree_id": "code"}, {"id": "b", "parent_id": "a", "tree_id": "context"}
        ]})


def test_localization_inspects_both_languages_and_stable_signature():
    good = {
        "signature": "def source_fn()",
        "title": {"zh": "运行源码", "en": "Run source"},
        "summary": {"zh": "根据 AST 构建源码证据", "en": "Build AST-backed evidence"},
        "description": {"zh": "保留原始签名", "en": "Preserves the original signature"},
    }
    assert invariants.localization_defects(good) == []
    broken = dict(good)
    broken["title"] = {"en": "Run source"}
    broken["localized_signature"] = "定义 source_fn()"
    issues = invariants.localization_defects(broken)
    assert "title.zh:missing" in issues
    assert "signature:changed" in issues
    broken["summary"] = {"en": "Run source."}
    assert "summary.en:tautology" in invariants.localization_defects(broken)
    assert "summary.zh:missing" in invariants.localization_defects(broken)
    broken["description"] = "not a mapping"
    assert "description.zh:missing" in invariants.localization_defects(broken)


def test_contract_receipt_does_not_self_certify_execution():
    selected = registry.resolve(PLAN, "H04")
    record = {
        "schema": "saebase.harness.contract_receipt.v1",
        "registry_version": registry.catalog()["version"],
        "contract_ids": selected["contract_ids"],
        "case_id": "H04",
        "context_paths": [],
        "edited_paths": [
            "harness/tests/wp18/runs/example/H04/implementation.py"
        ],
        "commands": [{"argv": ["python", "-m", "pytest"], "exit_code": 0}],
        "unresolved": [],
    }
    result = trace.validate_receipt(record, expected=selected["contract_ids"])
    assert result["verified_execution"] is False
    record["edited_paths"] = ["harness/core/agent_flow.py"]
    with pytest.raises(registry.ContractError):
        trace.validate_receipt(record, expected=selected["contract_ids"])


def test_cursor_shell_gate_allows_only_read_only_registry_queries():
    path = registry.ROOT / ".cursor/hooks/guard_shell.py"
    spec = importlib.util.spec_from_file_location("universal_guard_shell", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    query = "python -m harness.contracts.registry resolve --plan " + PLAN + " --case H04"
    assert module.gate(query) is None
    assert module.gate("python -m harness.contracts.registry list") is None
    assert module.gate("python -m harness.contracts.registry check --plan " + PLAN) is None
    assert module.gate(query + " && touch owned.py") is not None
    assert module.gate(
        "python -m harness.contracts.registry resolve --plan ../../other.md --case H04"
    ) is not None
