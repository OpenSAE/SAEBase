"""Source identity and provenance must survive public semantic compaction."""

import json
from pathlib import Path

from harness.code.presentation import readable_code_tree, SOURCE_REVIEWS
from harness.code.semantic_projection import compact_public_code_tree

ROOT = Path(__file__).resolve().parents[2]


def test_generated_atlas_keeps_all_source_interfaces():
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    graph = readable_code_tree(source)
    original_ids = {node["id"] for node in graph["nodes"]}
    projected = compact_public_code_tree(graph)
    retained = {node["id"] for node in projected["nodes"]}
    aliases = projected["source_api_index"]
    assert retained.isdisjoint(aliases)
    assert retained | set(aliases) == original_ids
    assert set(aliases.values()) <= retained
    assert SOURCE_REVIEWS.keys() <= retained
    assert len(projected["nodes"]) < len(original_ids)
    assert not any(
        "源码依据和调用参数" in str(node.get("summary") or "") for node in projected["nodes"]
    )
    by_id = {node["id"]: node for node in projected["nodes"]}
    for owner, node in by_id.items():
        blocks = [a for a in node.get("attributes") or [] if a["type"] == "source_interfaces"]
        if not blocks:
            continue
        records = [x for block in blocks for x in block["members"]]
        assert all(r["path"] or r["id"].startswith("code:") for r in records)
        assert len({r["id"] for r in records}) == len(records)
        for record in records:
            assert aliases[record["id"]] == owner


def test_compaction_is_stable_and_preserves_source_identity():
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    original = readable_code_tree(source)
    first = compact_public_code_tree(original)
    other = compact_public_code_tree(readable_code_tree(source))
    assert [n["id"] for n in first["nodes"]] == [n["id"] for n in other["nodes"]]
    assert first["source_api_index"] == other["source_api_index"]
    assert first["source_api_total"] == other["source_api_total"]


def test_concept_categories_and_reviewed_task_modules_survive():
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    graph = compact_public_code_tree(readable_code_tree(source))
    retained = {node["id"] for node in graph["nodes"]}
    for name in ("capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"):
        assert "code:instance:tasks:" + name in retained
        assert "code:module:tasks." + name + ".task" in retained
        assert "code:module:tasks." + name + ".operation" in retained
    assert sum(node.get("concept_group", False) for node in graph["nodes"]) >= 30
