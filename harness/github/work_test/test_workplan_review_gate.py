"""Negative checks for two independent Work Plan human-review deliverables."""

import copy

import pytest

from harness.github.work_plan import review_gate as gate


def example():
    plan = "# WP24\n" + "\n".join(f"## {i}. Section {i}\nBody" for i in range(1, 9))
    plan += '\n{"plan_id":"WP24"}\n'
    summary = "# WP24 · 人类审阅摘要\n## 概览\n" + "本次说明研究交付与人类审核范围" * 6
    for i in range(1, 9):
        summary += f"\n## {i}. Section {i}\n" + "明确本节任务和交付验收条件" * 7
    impact = {
        "schema": "saebase.workplan.node_impact.v1",
        "plan_id": "WP24",
        "source_commit": "a" * 40,
        "node_review_decision": "NO_NODE_CHANGE",
        "source_write_review": {"required": True, "status": "PENDING"},
        "domains": [
            {
                "domain": name,
                "decision": "NONE",
                "feature_effect": "UNCHANGED",
                "source_paths": ["harness/github/work_plan/WRITING_STANDARD.md"],
                "existing_nodes": [],
                "reason": "Current source is used as a reference; lasting node responsibility and attributes do not change.",
                "proposed_fields": {},
            }
            for name in sorted(gate.DOMAINS)
        ],
    }
    return plan, summary, impact


def test_valid_summary_and_no_node_change():
    plan, summary, impact = example()
    assert gate._parse_plan(plan) == ("WP24", list(range(1, 9)))
    assert gate.check_summary(summary, "WP24", list(range(1, 9)))["total_characters"] > 400
    assert gate.check_impact(impact, "WP24")["node_review_decision"] == "NO_NODE_CHANGE"


def test_missing_section_is_blocked():
    _, summary, _ = example()
    summary = summary.replace("## 4. Section 4", "### 4. Section 4")
    with pytest.raises(gate.ReviewGateError, match="match every"):
        gate.check_summary(summary, "WP24", list(range(1, 9)))


def test_short_summary_is_blocked():
    _, summary, _ = example()
    with pytest.raises(gate.ReviewGateError, match="Overview"):
        gate.check_summary(
            summary.replace("本次说明研究交付与人类审核范围" * 6, "摘要"), "WP24", list(range(1, 9))
        )


def test_reference_only_never_requires_node_mutation():
    _, _, audit = example()
    audit["domains"][0]["proposed_fields"] = {"description": "This WP changes nothing"}
    with pytest.raises(gate.ReviewGateError, match="NONE cannot"):
        gate.check_impact(audit, "WP24")


def test_specific_workplan_cannot_become_durable_node_text():
    _, _, audit = example()
    audit["node_review_decision"] = "NODE_HUMAN_CHECK"
    row = audit["domains"][0]
    row.update(
        decision="CHANGE",
        feature_effect="CHANGED",
        human_check_required=True,
        existing_nodes=["g.pipeline"],
        proposed_fields={"properties": "WP23 QA500 pilot"},
    )
    with pytest.raises(gate.ReviewGateError, match="specific"):
        gate.check_impact(audit, "WP24")


def test_changed_node_must_receive_human_check():
    _, _, audit = example()
    audit["node_review_decision"] = "NODE_HUMAN_CHECK"
    row = audit["domains"][0]
    row.update(
        decision="CHANGE",
        feature_effect="CHANGED",
        existing_nodes=["g.pipeline"],
        proposed_fields={"description": "A general new capability"},
    )
    with pytest.raises(gate.ReviewGateError, match="independent Human Check"):
        gate.check_impact(audit, "WP24")


def test_deferral_never_masquerades_as_no_change():
    _, _, audit = example()
    audit["domains"][0]["decision"] = "DEFER"
    audit["domains"][0]["feature_effect"] = "UNKNOWN"
    with pytest.raises(gate.ReviewGateError, match="node-review decision"):
        gate.check_impact(audit, "WP24")
    audit["node_review_decision"] = "DEFER_NODE_DECISION"
    assert gate.check_impact(audit, "WP24")["node_review_decision"] == "DEFER_NODE_DECISION"


def test_source_review_remains_separate():
    _, _, audit = example()
    audit["source_write_review"] = {"required": True, "status": "NOT_REQUIRED"}
    with pytest.raises(gate.ReviewGateError, match="Required source review"):
        gate.check_impact(audit, "WP24")


def test_no_duplicate_domains():
    _, _, audit = example()
    audit["domains"][1] = copy.deepcopy(audit["domains"][0])
    with pytest.raises(gate.ReviewGateError, match="repeated"):
        gate.check_impact(audit, "WP24")
