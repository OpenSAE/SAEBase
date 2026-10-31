"""Semantic Code tree acceptance: concepts describe work; AST remains authority."""

from collections import defaultdict
import json
from pathlib import Path

from harness.code.presentation import readable_code_tree
from harness.code.functional_taxonomy import is_reexport_only

ROOT = Path(__file__).resolve().parents[2]


def test_saebase_concepts_are_meaningful_and_lossless() -> None:
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    visible = readable_code_tree(source)
    originals = {node["id"] for node in source["nodes"]}
    by_id = {n["id"]: n for n in visible["nodes"]}
    children = defaultdict(list)
    for edge in visible["edges"]:
        if edge["type"] == "contains":
            children[edge["source"]].append(edge["target"])
    parent = "code:group:saebase"
    domains = children[parent]
    assert 10 <= len(domains) <= 20, len(domains)
    catalog = json.loads((ROOT / "harness/code/atlas_concepts.json").read_text(encoding="utf-8"))
    reviewed = {"code:concept:" + item["name"] for item in catalog["concepts"]}
    assert set(domains) == reviewed, "Concept domains must match the reviewed catalog"
    assert "code:concept:research_backends" not in reviewed
    assert len(set(by_id[x]["title"] for x in domains)) == len(domains)
    source_modules = {
        n["id"]
        for n in source["nodes"]
        if n.get("kind") == "module"
        and (n.get("path") or "").startswith("saebase/")
        and not n["path"].endswith("/__init__.py")
        and not is_reexport_only(n)
    }
    seen_modules = []
    queue = list(domains)
    generated = []
    while queue:
        node_id = queue.pop(0)
        node = by_id[node_id]
        if node.get("concept_group"):
            generated.append(node)
            assert node["context_term_ids"], node_id
            assert node["evidence"], node_id
            assert node["english_title"] and node["english_summary"], node_id
            assert node["title"] != node["summary"], node_id
            assert not any(
                word in node["title"]
                for word in ("第1组", "第2组", "相关操作", "其他接口", "源码职责浏览")
            ), node_id
            assert len(children[node_id]) <= 20, (node_id, len(children[node_id]))
            queue.extend(children[node_id])
        else:
            assert node["kind"] == "module", node_id
            seen_modules.append(node_id)
    assert set(seen_modules) == source_modules
    assert len(seen_modules) == len(source_modules), "Duplicated module assignment"
    assert len(generated) >= 30
    assert not any(is_reexport_only(n) for n in visible["nodes"])
    assert sum(is_reexport_only(n) for n in source["nodes"]) >= 24
    # The source tree is not deleted or rewritten by this presentation hierarchy.
    excluded = {n["id"] for n in source["nodes"] if is_reexport_only(n)}
    test_internals = {
        n["id"]
        for n in source["nodes"]
        if n.get("kind") in {"function", "method", "class"}
        and (n.get("path") or "").startswith("saebase/research_lab/tests/")
    }
    # The Atlas intentionally omits __init__, private helpers, and test
    # assertions. Every displayed implementation must still be an actual AST
    # entity; public module retention is checked above.
    displayed_source_ids = {n["id"] for n in visible["nodes"] if not n.get("presentation_only")}
    assert displayed_source_ids.issubset(originals)
    assert not displayed_source_ids.intersection(excluded | test_internals)
    assert visible["concept_hierarchy_schema"] == "saebase.atlas.concepts.v1"


def test_every_concept_has_distinct_sibling_responsibility() -> None:
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    graph = readable_code_tree(source)
    by_id = {node["id"]: node for node in graph["nodes"]}
    child_map = defaultdict(list)
    for edge in graph["edges"]:
        if edge["type"] == "contains":
            child_map[edge["source"]].append(edge["target"])
    for parent, children in child_map.items():
        unique = [by_id[key]["title"] for key in children if by_id[key].get("concept_group")]
        assert len(unique) == len(set(unique)), (parent, unique)
    actual = [n for n in graph["nodes"] if n.get("concept_group")]
    assert all(n["id"].startswith("code:concept:") for n in actual)
    assert not any(n["id"].startswith("code:nav:") for n in graph["nodes"])
    excessive = {node_id: len(items) for node_id, items in child_map.items() if len(items) > 20}
    assert not excessive, "Code Tree exceeds 20 children per node: " + repr(excessive)
