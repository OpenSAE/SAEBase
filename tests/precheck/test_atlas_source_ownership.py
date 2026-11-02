"""Check SAE ownership against a fresh source-derived AST, not stale snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from harness.code.build import build
from harness.code.functional_taxonomy import organize_saebase_concepts

ROOT = Path(__file__).resolve().parents[2]


def test_migrated_sae_code_tree_has_unambiguous_source_ownership():
    graph = build()
    paths = {n.get("path") for n in graph["nodes"] if n.get("kind") == "module"}
    assert not any(str(path).startswith("saebase/sae/") for path in paths)
    assert "saebase/api/sae_profiles.py" in paths
    assert "saebase/api/sae_loading.py" in paths
    assert "tasks/benchmark/activation_health.py" in paths
    result = organize_saebase_concepts(graph)
    assert result["concepts"] >= 10
    assert result["source_modules"] >= 20


def test_reviewed_sae_semantics_use_real_migrated_source_paths():
    spec = json.loads((ROOT / "harness/code/reviewed_semantics.json").read_text(encoding="utf-8"))
    for node_id, details in spec["nodes"].items():
        assert not node_id.startswith(("code:module:saebase.sae.", "code:method:saebase.sae."))
        assert not str(details.get("source_path", "")).startswith("saebase/sae/")


def test_live_atlas_models_use_family_series_model_hierarchy():
    from harness.code.presentation import readable_code_tree

    view = readable_code_tree(build())
    ids = {node["id"] for node in view["nodes"]}
    for item in (
        "code:concrete:models",
        "code:family:models:gemma",
        "code:family:models:gemma:gemma2",
        "code:instance:models:gemma2_2b_base",
        "code:family:models:qwen:qwen35",
        "code:instance:models:qwen35_2b",
        "code:legacy:models",
    ):
        assert item in ids
    edges = {(edge["source"], edge["target"]) for edge in view["edges"]}
    assert (
        "code:family:models:gemma:gemma2",
        "code:instance:models:gemma2_2b_base",
    ) in edges
