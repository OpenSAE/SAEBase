"""Regressions for generated Atlas semantic collisions."""
from harness.code.audit_semantics import audit, purpose, similarity


def test_same_titles_descriptions_but_different_symbols_are_detected():
    parent = {"id": "parent", "title": "SAE 验证", "summary": "验证任务"}
    nodes = [parent]
    for i in range(4):
        nodes.append({
            "id": f"fn{i}", "kind": "method", "source_symbol": f"method_{i}",
            "path": f"saebase/module_{i}.py",
            "title": "管理SAE 研究能力验证与执行",
            "summary": "负责管理SAE 研究能力验证与执行；源码依据和调用参数可在节点的展开说明中查看。",
        })
    tree = {"code": {
        "nodes": nodes,
        "edges": [{"type": "contains", "source": "parent", "target": n["id"]} for n in nodes[1:]],
    }}
    result = audit(tree)
    for field in ("repeated_sibling_title", "repeated_sibling_summary", "similar_sibling_purpose"):
        assert result["counts"][field] == 1
        assert result["issues"][field][0]["size"] == 4
        assert len(result["issues"][field][0]["members"]) == 4
    assert result["counts"]["template_summary"] == 4
    assert result["counts"]["tautological_purpose"] == 4


def test_verb_swaps_cannot_hide_repeated_research_objects():
    tree = {"code": {"nodes": [
        {"id": "root", "title": "研究系统"},
        {"id": "a", "title": "实现SAE 研究能力验证与执行", "summary": "完成SAE 研究能力验证与执行"},
        {"id": "b", "title": "管理SAE 研究能力验证与执行", "summary": "负责管理SAE 研究能力验证与执行"},
    ], "edges": [
        {"source": "root", "target": "a", "type": "contains"},
        {"source": "root", "target": "b", "type": "contains"},
    ]}}
    result = audit(tree)
    assert result["counts"]["repeated_sibling_title"] == 0
    assert result["counts"]["similar_sibling_purpose"] == 1


def test_no_collision_for_distinct_scientific_operations():
    tree = {"code": {"nodes": [
        {"id": "root", "title": "研究系统"},
        {"id": "a", "title": "计算SAE重建损失", "summary": "比较残差重建前后的均方误差"},
        {"id": "b", "title": "测量特征干预效果", "summary": "比较特征消融前后的目标Token概率"},
    ], "edges": [
        {"source": "root", "target": "a", "type": "contains"},
        {"source": "root", "target": "b", "type": "contains"},
    ]}}
    result = audit(tree)
    assert result["counts"].get("similar_sibling_purpose", 0) == 0


def test_context_parent_map_and_knowledge_edges_are_scanned():
    tree = {
        "context": {"nodes": [
            {"id": "ctx", "title": "术语"},
            {"id": "x", "title": "管理SAE执行", "summary": "说明SAE执行"},
            {"id": "y", "title": "管理SAE执行", "summary": "说明SAE执行"},
        ], "parents": {"x": "ctx", "y": "ctx"}},
        "knowledge": {"nodes": [
            {"id": "k", "title": "科研"},
            {"id": "a", "title": "实施验证研究", "summary": "验证研究"},
            {"id": "b", "title": "实施验证研究", "summary": "验证研究"},
        ], "edges": [
            {"source": "k", "target": "a", "type": "decomposes"},
            {"source": "k", "target": "b", "type": "decomposes"},
        ]},
    }
    result = audit(tree)
    assert {r["tree"] for r in result["issues"]["repeated_sibling_title"]} == {"context", "knowledge"}


def test_purpose_removes_template_without_adding_symbol_identity():
    assert purpose("负责管理SAE 研究能力验证与执行；源码依据和调用参数可在节点的展开说明中查看。") == purpose("实现SAE 研究能力验证与执行")
    assert similarity("捕获SAE特征激活并计算重建", "捕获SAE特征激活并计算重建误差") > .82
