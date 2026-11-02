"""Negative checks for two independent Work Plan human-review deliverables."""

import copy
import json

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
        "human_check_review": {"status": "NOT_REQUIRED", "reason": "Every observed durable Atlas node remains semantically unchanged; therefore no Node Human Check is necessary."},
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
    audit["human_check_review"] = {"status": "NOT_REQUIRED_NOW", "reason": "The proposed module has not been implemented, so inspect actual AST and ownership after implementation.", "recheck_after_implementation": True}
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


def test_versioned_workplan_lifecycle_enforces_dual_review(tmp_path):
    from harness.github.work_plan import lifecycle as life

    base_plan, summary, impact = example()

    def put(name, payload):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )

    put("harness/core/project_graph.json", {"nodes": [{"id": "g.pipeline"}]})
    put("doc/knowledge/context_graph.json", {"nodes": [{"id": "q.sae.methods"}]})
    put("doc/knowledge/glossary.json", {"terms": [{"id": "term.sae"}]})
    put("doc/knowledge/atlas_terms.json", {"terms": []})
    put("saebase/runner.py", "def run(): pass\n")
    put("conf/experiment/demo.yaml", "experiment:\n  name: demo\n")
    intent = {
        "schema": "saebase.from_human.v1",
        "plan_id": "WP24",
        "goal": "Deliver source-backed review",
        "scientific_motivation": "Reproducibility",
    }
    fence = chr(96) * 3
    put(
        "harness/github/work_plan/from_human/WP24.md",
        fence + "from-human-json\n" + json.dumps(intent) + "\n" + fence + "\n",
    )
    obj = {
        "schema": "saebase.workplan.v1",
        "plan_id": "WP24",
        "status": "draft",
        "from_human": "harness/github/work_plan/from_human/WP24.md",
        "atlas": {
            "core_ids": ["g.pipeline"],
            "knowledge_ids": ["q.sae.methods"],
            "context_term_ids": ["term.sae"],
            "code_refs": ["saebase/runner.py"],
        },
        "experiment": {"study_id": "demo", "config": "conf/experiment/demo.yaml"},
        "steps": [
            {
                "id": "setup",
                "needs": [],
                "permission": "local",
                "goal": "Check source",
                "evidence": "source receipt",
            }
        ],
        "deliverables": [{"experiment": "E1", "metric": "source checks", "expected": "one"}],
    }
    path = "harness/github/work_plan/0.6.5_wp024_review_gate.md"
    put(path, base_plan + "\n" + fence + "workplan-json\n" + json.dumps(obj) + "\n" + fence + "\n")
    with pytest.raises(FileNotFoundError):
        life.check(path, root=tmp_path, check_contracts=False)
    put("harness/github/work_plan/reviews/WP24/summary.md", summary)
    put("harness/github/work_plan/reviews/WP24/node_impact.json", impact)
    assert life.check(path, root=tmp_path, check_contracts=False)["plan_id"] == "WP24"


def test_missing_human_check_disposition_is_blocked():
    _, _, impact = example()
    del impact["human_check_review"]
    with pytest.raises(gate.ReviewGateError, match="Human Check decision"):
        gate.check_impact(impact, "WP24")


def test_no_node_check_requires_specific_reason():
    _, _, impact = example()
    impact["human_check_review"]["reason"] = "No."
    with pytest.raises(gate.ReviewGateError, match="source-backed explanation"):
        gate.check_impact(impact, "WP24")


def test_deferral_must_require_ast_recheck():
    _, _, impact = example()
    item = impact["domains"][0]
    item["decision"] = "DEFER"
    item["feature_effect"] = "UNKNOWN"
    impact["node_review_decision"] = "DEFER_NODE_DECISION"
    impact["human_check_review"] = {
        "status": "NOT_REQUIRED_NOW",
        "reason": "A proposed new Code module has not yet been implemented and must be audited after its real AST exists.",
        "recheck_after_implementation": False,
    }
    with pytest.raises(gate.ReviewGateError, match="AST recheck"):
        gate.check_impact(impact, "WP24")


def test_real_node_edit_needs_atlas_url():
    _, _, impact = example()
    item = impact["domains"][0]
    item["decision"] = "CHANGE"
    item["feature_effect"] = "CHANGED"
    item["existing_nodes"] = ["g.pipeline"]
    item["human_check_required"] = True
    item["proposed_fields"] = {"description": "An independently evidenced reusable capability"}
    impact["node_review_decision"] = "NODE_HUMAN_CHECK"
    impact["human_check_review"] = {
        "status": "REQUIRED",
        "reason": "The existing node has a concrete durable responsibility change, requiring Atlas Human Check before edits.",
    }
    with pytest.raises(gate.ReviewGateError, match="Atlas review URL"):
        gate.check_impact(impact, "WP24")
    impact["human_check_review"]["review_url"] = (
        "https://atheavenma.github.io/saebase/atlas/core.html?review=HC-Example"
    )
    assert gate.check_impact(impact, "WP24")["human_check_review"] == "REQUIRED"
