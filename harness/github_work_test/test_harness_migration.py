"""Regression checks for the unified repository's reviewed Harness paths."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_evidence_targets_exist() -> None:
    graph = json.loads((ROOT / "harness/core/project_graph.json").read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        for path in node.get("evidence", []):
            assert (ROOT / path).is_file(), (node["id"], path)


def test_agent_plan_storage_matches_harness_structure() -> None:
    from harness.core.agent_flow import PLANS

    assert PLANS == ROOT / "harness/github_work_plan/agent_flow"
    for path in (
        "harness/core/agent_flow.py",
        "harness/core/github_review.py",
        ".cursor/skills/workflow/run-work-order/SKILL.md",
    ):
        assert "harness/github_harness/" not in (ROOT / path).read_text(encoding="utf-8")


def test_legacy_component_directories_removed() -> None:
    for directory in ("dlbase", "research_datasets", "apps", "scripts", "examples", "work_plan"):
        assert not (ROOT / directory).exists(), directory
