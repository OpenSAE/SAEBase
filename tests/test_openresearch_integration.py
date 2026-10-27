"""Offline structural checks for additive OpenResearch integration."""

from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]


def test_original_saebase_public_entrypoints_preserved():
    for name in (
        "saebase/cli.py",
        "saebase/api/model.py",
        "saebase/data/__init__.py",
        "models/plugins.py",
        "tasks/plugins.py",
        "conf/config.yaml",
        "pyproject.toml",
    ):
        assert (ROOT / name).is_file(), name


def test_openresearch_layers_are_additive():
    for name in (
        "dlbase/train.py",
        "dlbase/registry.py",
        "research_datasets/plugins.py",
        "research_datasets/NLP/artifacts.py",
        "models/dlbase_plugins.py",
        "tasks/dlbase_plugins.py",
        "models/LLM/gemma2_2b_base/gemma2_2b_base.py",
        "conf/openresearch.yaml",
        "harness/core/agent_flow.py",
        "harness/code/build.py",
        "harness/visualization/i18n.js",
        ".cursor/hooks/guard_paths.py",
        ".cursor/hooks/guard_shell.py",
    ):
        assert (ROOT / name).is_file(), name


def test_seven_llm_tasks_present():
    for task in ("capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"):
        assert (ROOT / "tasks" / "LLM" / task / "task.py").exists(), task


def test_isolated_registries_preserve_old_namespaces():
    old_models = (ROOT / "models/plugins.py").read_text(encoding="utf-8")
    old_tasks = (ROOT / "tasks/plugins.py").read_text(encoding="utf-8")
    assert "saebase.model_plugins" in old_models
    assert "saebase.tasks" in old_tasks
    new_bootstrap = (ROOT / "dlbase/plugin_bootstrap.py").read_text(encoding="utf-8")
    assert "models.dlbase_plugins" in new_bootstrap
    assert "tasks.dlbase_plugins" in new_bootstrap
    training = (ROOT / "dlbase/train.py").read_text(encoding="utf-8")
    assert 'config_name="openresearch"' in training


def test_graph_and_policy_parse():
    for file in (
        "harness/core/project_graph.json",
        ".cursor/policy.json",
        "doc/knowledge/context_graph.json",
    ):
        assert isinstance(json.loads((ROOT / file).read_text(encoding="utf-8")), dict)


def test_python_syntax_in_added_modules():
    for root in ("dlbase", "research_datasets", "harness"):
        for file in (ROOT / root).rglob("*.py"):
            ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
