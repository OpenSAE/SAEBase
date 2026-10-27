"""Import / package boundary tests for SAEBase (no model download)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = ROOT / "tasks" / "sae"
FORBIDDEN_TOP = {
    "dlbase",
    "pytorch_lightning",
    "torchvision",
    "cv2",
}


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    return mods


def test_package_layout():
    assert SAE_DIR.is_dir()
    assert not (ROOT / "datasets").exists()
    assert (ROOT / "saebase").is_dir()
    assert (ROOT / "dlbase" / "registry.py").is_file()
    assert (ROOT / "research_datasets" / "plugins.py").is_file()
    assert not (ROOT / "tasks" / "SAR").exists()
    assert not (ROOT / "tasks" / "sar").exists()
    assert not (ROOT / "tasks" / "SAE").exists()
    assert not (ROOT / "tasks" / "classification").exists()
    assert not (ROOT / "saebase" / "data" / "CUB_200").exists()
    assert not (ROOT / "saebase" / "data" / "I_RAVEN").exists()
    assert not (ROOT / "models" / "resnet18_mlp").exists()


def test_no_forbidden_imports_in_active_packages():
    roots = [ROOT / "saebase", ROOT / "tasks" / "sae", ROOT / "models", ROOT / "saebase" / "data"]
    for root in roots:
        for path in root.rglob("*.py"):
            if root == ROOT / "models" and (
                path.name in {"dlbase_plugins.py", "build.py"}
                or path.relative_to(root).parts[0] in {"LLM", "vision"}
            ):
                continue  # OpenResearch-only model namespace uses its own DLBase contract
            if "__pycache__" in path.parts:
                continue
            for mod in _collect_imports(path):
                top = mod.split(".")[0]
                assert top not in FORBIDDEN_TOP, f"{path} imports forbidden {mod}"
                assert not mod.startswith("tasks.SAR")
                assert not mod.startswith("tasks.sar.")
                assert not mod.startswith("tasks.SAE")
                assert not mod.startswith("dlbase")


def test_sae_plugin_importable():
    import tasks.plugins  # noqa: F401
    from saebase.registry import get_task_plugin
    from saebase.sae.task import SAETask

    task = SAETask()
    assert task.name == "sae"
    plugin = get_task_plugin("sae")
    assert plugin.task_class is SAETask
