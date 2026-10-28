"""Five-tree semantics, source-preserving interface consolidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.code.presentation import readable_code_tree
from harness.visualization.semantic_quality import validate_all_trees

ROOT = Path(__file__).resolve().parents[2]


def fixture_graphs() -> dict:
    terms = {
        "nodes": [
            {
                "id": "term.sae",
                "title": "SAE",
                "summary": "稀疏字典",
                "description": "用特征表示模型激活。",
            }
        ],
        "edges": [],
    }
    other = {
        "nodes": [
            {
                "id": "x",
                "title": "采集模型表示",
                "summary": "读取指定 Hook 的真实激活向量。",
                "description": "读取指定 Hook 的真实激活向量。",
            }
        ],
        "edges": [],
    }
    return {
        "core": json.loads(json.dumps(other)),
        "knowledge": json.loads(json.dumps(other)),
        "code": json.loads(json.dumps(other)),
        "environment": json.loads(json.dumps(other)),
        "context": terms,
    }


def test_all_five_trees_reject_same_title_and_summary() -> None:
    for view in ("core", "knowledge", "code", "environment", "context"):
        graphs = fixture_graphs()
        node = graphs[view]["nodes"][0]
        node["summary"] = node["title"] + "。"
        with pytest.raises(ValueError, match="title equals card summary"):
            validate_all_trees(graphs)


def test_all_five_trees_allow_distinct_source_prose() -> None:
    result = validate_all_trees(fixture_graphs())
    assert set(result) == {"core", "knowledge", "code", "environment", "context"}


def test_unreviewed_helpers_retain_full_source_interfaces() -> None:
    original = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    tree = readable_code_tree(original)
    by_id = {node["id"]: node for node in tree["nodes"]}
    folded = tree.get("folded_interface_count", 0)
    assert folded > 100, "Generic source helpers were not consolidated"
    preserved = set()
    for node in tree["nodes"]:
        for attr in node.get("attributes") or []:
            if attr.get("type") == "class_methods":
                for method in attr.get("methods") or []:
                    if method.get("symbol_id"):
                        assert method.get("signature") and method.get("source_path")
                        preserved.add(method["symbol_id"])
    assert len(preserved) > 50
    for source in preserved:
        assert source not in by_id, "Merged helper remains a duplicate Tree node"
    assert "code:function:saebase.registry:register_dataset" in by_id
