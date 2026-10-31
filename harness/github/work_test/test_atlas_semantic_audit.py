"""Regressions for generated Atlas semantic collisions."""

from harness.code.audit_semantics import audit, purpose, similarity


def test_same_titles_descriptions_but_different_symbols_are_detected():
    parent = {"id": "parent", "title": "SAE 验证", "summary": "验证任务"}
    nodes = [parent]
    for i in range(4):
        nodes.append(
            {
                "id": f"fn{i}",
                "kind": "method",
                "source_symbol": f"method_{i}",
                "path": f"saebase/module_{i}.py",
                "title": "管理SAE 研究能力验证与执行",
                "summary": "负责管理SAE 研究能力验证与执行；源码依据和调用参数可在节点的展开说明中查看。",
            }
        )
    tree = {
        "code": {
            "nodes": nodes,
            "edges": [
                {"type": "contains", "source": "parent", "target": n["id"]} for n in nodes[1:]
            ],
        }
    }
    result = audit(tree)
    for field in ("repeated_sibling_title", "repeated_sibling_summary", "similar_sibling_purpose"):
        assert result["counts"][field] == 1
        assert result["issues"][field][0]["size"] == 4
        assert len(result["issues"][field][0]["members"]) == 4
    assert result["counts"]["template_summary"] == 4
    assert result["counts"]["tautological_purpose"] == 4


def test_verb_swaps_cannot_hide_repeated_research_objects():
    tree = {
        "code": {
            "nodes": [
                {"id": "root", "title": "研究系统"},
                {
                    "id": "a",
                    "title": "实现SAE 研究能力验证与执行",
                    "summary": "完成SAE 研究能力验证与执行",
                },
                {
                    "id": "b",
                    "title": "管理SAE 研究能力验证与执行",
                    "summary": "负责管理SAE 研究能力验证与执行",
                },
            ],
            "edges": [
                {"source": "root", "target": "a", "type": "contains"},
                {"source": "root", "target": "b", "type": "contains"},
            ],
        }
    }
    result = audit(tree)
    assert result["counts"].get("repeated_sibling_title", 0) == 0
    assert result["counts"]["similar_sibling_purpose"] == 1


def test_no_collision_for_distinct_scientific_operations():
    tree = {
        "code": {
            "nodes": [
                {"id": "root", "title": "研究系统"},
                {"id": "a", "title": "计算SAE重建损失", "summary": "比较残差重建前后的均方误差"},
                {
                    "id": "b",
                    "title": "测量特征干预效果",
                    "summary": "比较特征消融前后的目标Token概率",
                },
            ],
            "edges": [
                {"source": "root", "target": "a", "type": "contains"},
                {"source": "root", "target": "b", "type": "contains"},
            ],
        }
    }
    result = audit(tree)
    assert result["counts"].get("similar_sibling_purpose", 0) == 0


def test_context_parent_map_and_knowledge_edges_are_scanned():
    tree = {
        "context": {
            "nodes": [
                {"id": "ctx", "title": "术语"},
                {"id": "x", "title": "管理SAE执行", "summary": "说明SAE执行"},
                {"id": "y", "title": "管理SAE执行", "summary": "说明SAE执行"},
            ],
            "parents": {"x": "ctx", "y": "ctx"},
        },
        "knowledge": {
            "nodes": [
                {"id": "k", "title": "科研"},
                {"id": "a", "title": "实施验证研究", "summary": "验证研究"},
                {"id": "b", "title": "实施验证研究", "summary": "验证研究"},
            ],
            "edges": [
                {"source": "k", "target": "a", "type": "decomposes"},
                {"source": "k", "target": "b", "type": "decomposes"},
            ],
        },
    }
    result = audit(tree)
    assert {r["tree"] for r in result["issues"]["repeated_sibling_title"]} == {
        "context",
        "knowledge",
    }


def test_purpose_removes_template_without_adding_symbol_identity():
    assert purpose(
        "负责管理SAE 研究能力验证与执行；源码依据和调用参数可在节点的展开说明中查看。"
    ) == purpose("实现SAE 研究能力验证与执行")
    assert similarity("捕获SAE特征激活并计算重建", "捕获SAE特征激活并计算重建误差") > 0.82

def test_cross_branch_duplicate_roles_are_detected() -> None:
    """Two separately nested source symbols with one actual purpose must fail."""
    tree = {
        "code": {
            "nodes": [
                {"id": "root", "title": "SAE 科研能力", "summary": "组织科研方法"},
                {"id": "p1", "title": "研究模块 A", "summary": "实验分组 A"},
                {"id": "p2", "title": "研究模块 B", "summary": "实验分组 B"},
                {
                    "id": "a",
                    "kind": "method",
                    "title": "实现稀疏特征研究软件与运行接口",
                    "summary": "负责实现稀疏特征研究软件与运行接口",
                    "path": "saebase/a.py",
                },
                {
                    "id": "b",
                    "kind": "method",
                    "title": "管理稀疏特征研究软件与运行接口",
                    "summary": "负责管理稀疏特征研究软件与运行接口",
                    "path": "saebase/b.py",
                },
            ],
            "edges": [
                {"type": "contains", "source": "root", "target": "p1"},
                {"type": "contains", "source": "root", "target": "p2"},
                {"type": "contains", "source": "p1", "target": "a"},
                {"type": "contains", "source": "p2", "target": "b"},
            ],
        }
    }
    report = audit(tree)
    assert report["counts"].get("repeated_sibling_title", 0) == 0
    assert report["counts"]["similar_global_purpose"] == 1
    assert {item["parent_id"] for item in report["issues"]["similar_global_purpose"][0]["members"]} == {"p1", "p2"}


def test_cross_branch_identical_titles_and_descriptions_fail() -> None:
    graph = {
        "code": {
            "nodes": [
                {"id": "root", "title": "科研系统"},
                {"id": "p", "title": "训练", "summary": "训练运行"},
                {"id": "q", "title": "评测", "summary": "评测运行"},
                {"id": "a", "title": "处理模型运行", "summary": "执行具体模型分析"},
                {"id": "b", "title": "处理模型运行", "summary": "执行具体模型分析"},
            ],
            "edges": [
                {"type": "contains", "source": "root", "target": "p"},
                {"type": "contains", "source": "root", "target": "q"},
                {"type": "contains", "source": "p", "target": "a"},
                {"type": "contains", "source": "q", "target": "b"},
            ],
        }
    }
    report = audit(graph)
    assert report["counts"]["repeated_global_title"] == 1
    assert report["counts"]["repeated_global_summary"] == 1
    assert {node["id"] for node in report["issues"]["repeated_global_title"][0]["members"]} == {"a", "b"}
