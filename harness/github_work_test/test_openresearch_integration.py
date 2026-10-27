"""Offline structural checks for additive OpenResearch integration."""

from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[2]


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
        "saebase/runtime/train.py",
        "saebase/registry.py",
        "datasets/plugins.py",
        "datasets/artifacts.py",
        "models/plugins.py",
        "tasks/plugins.py",
        "models/gemma2_2b_base/gemma2_2b_base.py",
        "conf/sae_research.yaml",
        "harness/core/agent_flow.py",
        "harness/code/build.py",
        "harness/visualization/i18n.js",
        ".cursor/hooks/guard_paths.py",
        ".cursor/hooks/guard_shell.py",
    ):
        assert (ROOT / name).is_file(), name


def test_seven_llm_tasks_present():
    for task in ("capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"):
        assert (ROOT / "tasks" / task / "task.py").exists(), task


def test_unified_registry_and_hydra_entrypoint():
    registry = (ROOT / "saebase/registry.py").read_text(encoding="utf-8")
    assert "def register_dataset(" in registry
    assert "def register_model(" in registry
    assert "def register_task(" in registry
    models = (ROOT / "models/plugins.py").read_text(encoding="utf-8")
    tasks = (ROOT / "tasks/plugins.py").read_text(encoding="utf-8")
    assert "register_model" in models and "register_sae" in models
    assert "CaptureTask" in tasks and "TrainTask" in tasks
    bootstrap = (ROOT / "saebase/plugin_bootstrap.py").read_text(encoding="utf-8")
    assert "models.plugins" in bootstrap
    assert "tasks.plugins" in bootstrap
    assert "saebase.datasets.plugins" in bootstrap
    training = (ROOT / "saebase/runtime/train.py").read_text(encoding="utf-8")
    assert 'config_name="sae_research"' in training
    for retired in ("dlbase", "research_datasets", "tasks/LLM"):
        assert not (ROOT / retired).exists()



def test_graph_and_policy_parse():
    for file in (
        "harness/core/project_graph.json",
        ".cursor/policy.json",
        "doc/knowledge/context_graph.json",
    ):
        assert isinstance(json.loads((ROOT / file).read_text(encoding="utf-8")), dict)


def test_python_syntax_in_added_modules():
    for root in ("saebase/runtime", "datasets", "harness"):
        for file in (ROOT / root).rglob("*.py"):
            ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
