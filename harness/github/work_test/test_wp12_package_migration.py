"""Migration guard: WP12 implementations belong to stable functional packages."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MOVED = {
    "saebase/release/capability_acceptance.py",
    "saebase/benchmarks/native_comparisons.py",
    "saebase/benchmarks/adapters/custom_saebench.py",
    "saebase/circuits/graph.py",
    "saebase/circuits/transcoder_intervention.py",
    "saebase/circuits/clt_intervention.py",
    "saebase/interventions/research_edits.py",
    "saebase/interventions/ndif_adapter.py",
    "saebase/interventions/pyvene_adapter.py",
    "tasks/train/external_training.py",
}


def test_migrated_modules_compile_without_a_wp12_package():
    assert not (ROOT / "saebase/wp12").exists()
    assert len(MOVED) == 10
    for name in sorted(MOVED):
        file = ROOT / name
        assert file.is_file(), name
        compile(file.read_text(encoding="utf-8"), str(file), "exec")


def test_no_runtime_import_points_to_legacy_package():
    for file in (ROOT / "saebase").rglob("*.py"):
        ast_tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("saebase.wp12"), str(file)
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("saebase.wp12") for alias in node.names), str(
                    file
                )


def test_atlas_source_ownership_uses_only_real_functional_directories():
    spec = json.loads((ROOT / "harness/code/atlas_concepts.json").read_text(encoding="utf-8"))
    roots = spec["concepts"]
    assert not any(node["name"] == "research_backends" for node in roots)
    assert all("wp12" not in node["source_dirs"] for node in roots)
    interventions = next(node for node in roots if node["name"] == "interventions")
    assert set(interventions["source_dirs"]) == {"interventions", "circuits"}
    titles = (ROOT / "harness/code/semantic_titles.py").read_text(encoding="utf-8")
    assert '"saebase/wp12/"' not in titles


def test_committed_code_snapshot_has_only_migrated_source_paths():
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    paths = {node["path"] for node in source["nodes"] if node.get("path")}
    assert not any(path.startswith("saebase/wp12/") for path in paths)
    assert MOVED <= paths
    assert "saebase/research_lab/backend/published_records.py" in paths
    assert "saebase/research_lab/backend/published_wp12.py" not in paths


def test_published_records_integrations_use_new_module_name():
    """Historical data storage stays stable, but imports target the domain module."""
    roots = (
        ROOT / "saebase/research_lab/backend",
        ROOT / "saebase/research_lab/tests",
        ROOT / "saebase/tools",
    )
    for folder in roots:
        for file in folder.rglob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert all(
                        alias.name != "published_wp12" for alias in node.names
                    ), str(file)
                if isinstance(node, ast.Import):
                    assert all(
                        not alias.name.endswith(".published_wp12") for alias in node.names
                    ), str(file)
    published = ROOT / "saebase/research_lab/backend/published_records.py"
    source = published.read_text(encoding="utf-8")
    assert ' / "published_wp12"' in source, "Keep existing public evidence storage"

