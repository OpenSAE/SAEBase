"""Regression checks for the unified repository's reviewed Harness paths."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_core_evidence_targets_exist() -> None:
    graph = json.loads((ROOT / "harness/core/project_graph.json").read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        for path in node.get("evidence", []):
            assert (ROOT / path).is_file(), (node["id"], path)


def test_agent_plan_storage_matches_harness_structure() -> None:
    from harness.core.agent_flow import PLANS

    assert PLANS == ROOT / "harness/github/work_plan/agent_flow"
    for path in (
        "harness/core/agent_flow.py",
        "harness/core/github_review.py",
        ".cursor/skills/workflow/run-work-order/SKILL.md",
    ):
        assert "harness/github_harness/" not in (ROOT / path).read_text(encoding="utf-8")


def test_legacy_component_directories_removed() -> None:
    for directory in ("dlbase", "research_datasets", "apps", "examples", "work_plan"):
        assert not (ROOT / directory).exists(), directory


def test_code_navigation_requires_real_source_and_context() -> None:
    """No ordinal, role-guess, or source-range navigation pseudo-components."""
    from harness.code.presentation import readable_code_tree

    graph = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    visible = readable_code_tree(graph)
    originals = {n["id"] for n in graph["nodes"]}
    for node in visible["nodes"]:
        assert not node["id"].startswith("code:nav:"), node["id"]
        assert "第1组" not in node["title"] and "第2组" not in node["title"]
        if node.get("presentation_only"):
            assert node.get("context_term_ids"), node["id"]
        else:
            assert node["id"] in originals, node["id"]
    for edge in visible["edges"]:
        assert edge["source"] != edge["target"]
