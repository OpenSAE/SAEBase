"""Import and package boundaries for the unified SAEBase layout (no model download)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK_NAMES = ("capture", "intervene", "generate", "circuits", "explain", "benchmark", "train")
FORBIDDEN_TOP = {"dlbase", "torchvision", "cv2"}


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_package_layout():
    for directory in ("datasets", "models", "tasks", "saebase", "harness", "conf"):
        assert (ROOT / directory).is_dir(), directory
    for directory in ("dlbase", "research_datasets", "apps", "examples", "scripts", "work_plan"):
        assert not (ROOT / directory).exists(), directory
    assert (ROOT / "datasets" / "plugins.py").is_file()
    assert (ROOT / "saebase" / "registry.py").is_file()
    assert (ROOT / "saebase" / "runtime" / "train.py").is_file()
    for name in TASK_NAMES:
        assert (ROOT / "tasks" / name / "task.py").is_file(), name
    assert (ROOT / "tasks" / "sae" / "__init__.py").exists()
    assert (ROOT / "saebase" / "data" / "prompt_store.py").is_file()
    for retired in ("tasks/SAR", "tasks/sar", "tasks/SAE", "tasks/classification",
                    "saebase/data/CUB_200", "saebase/data/I_RAVEN"):
        assert not (ROOT / retired).exists(), retired


def test_no_forbidden_imports_in_active_packages():
    roots = [ROOT / name for name in ("saebase", "tasks", "models", "datasets")]
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for module in _collect_imports(path):
                top = module.split(".")[0]
                assert top not in FORBIDDEN_TOP, f"{path} imports forbidden {module}"
                # Lightning is an optional, training-only dependency.
                if top == "pytorch_lightning":
                    training_only = (
                        path.is_relative_to(ROOT / "saebase" / "runtime" / "training")
                        or path.is_relative_to(ROOT / "tasks" / "train")
                        or path == ROOT / "datasets" / "datamodule.py"
                    )
                    assert training_only, f"{path} eagerly imports optional {module}"
                assert not module.startswith(("tasks.SAR", "tasks.sar.", "tasks.SAE"))


def test_sae_plugin_importable():
    from saebase.registry import get_task_plugin
    from saebase.sae.task import SAETask

    task = SAETask()
    assert task.name == "sae"
    assert get_task_plugin("sae").task_class is SAETask
    for name in TASK_NAMES:
        assert get_task_plugin(name).task_class is not None
