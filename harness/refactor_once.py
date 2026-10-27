#!/usr/bin/env python3
"""One-time SAEBase source-tree consolidation using OpenResearch's component boundaries."""
from __future__ import annotations

import ast
import posixpath
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = ROOT / "saebase/data"
DATA = ROOT / "datasets"
TEXT_SUFFIX = {".py", ".md", ".json", ".yaml", ".yml", ".mdc", ".toml", ".js",
               ".html", ".css", ".sh", ".txt", ".cff", ".ini", ".cfg"}
SKIP_PREFIX = ("conf/coverage/stage_d_evidence/", ".git/", "harness/refactor_once.py")

def file(path: str) -> Path:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def put(path: str, content: str) -> None:
    file(path).write_text(content, encoding="utf-8")

def delete(path: str) -> None:
    target = ROOT / path
    if target.is_file():
        target.unlink()

def move(source: str, target: str) -> None:
    src, dst = ROOT / source, file(target)
    if not src.exists():
        return
    if dst.exists():
        raise FileExistsError(f"Destination not empty: {target}")
    shutil.move(str(src), str(dst))

def new_path(path: str) -> str:
    for a, b in (
        ("apps/research-lab/", "saebase/research_lab/"),
        ("docs/", "harness/github_development_doc/"),
        ("work_plan/", "harness/github_work_plan/"),
        ("scripts/", "saebase/tools/"),
        ("examples/", "saebase/examples/"),
        ("assets/", "saebase/assets/"),
        ("tasks/LLM/", "tasks/"),
        ("research_datasets/NLP/", "datasets/"),
        ("research_datasets/", "datasets/"),
        ("models/LLM/gemma2_2b_base/", "models/gemma2_2b_base/"),
        ("dlbase/", "saebase/runtime/"),
    ):
        if path.startswith(a):
            return b + path[len(a):]
    return path

def fix_markdown_links(content: str, old_src: str, new_src: str) -> str:
    def link(m):
        ref = m.group(2)
        if (not ref or ref.startswith(("#", "/", "http:", "https:", "mailto:", "data:"))
                or "://" in ref):
            return m.group(0)
        target, marker, anchor = ref.partition("#")
        if not target:
            return m.group(0)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(old_src), target))
        transformed = new_path(resolved)
        relative = posixpath.relpath(transformed, posixpath.dirname(new_src))
        return m.group(1) + relative + (marker + anchor if marker else "") + m.group(3)
    return re.sub(r"(\]\()([^\s\)]+)(\))", link, content)

def move_project_resources() -> None:
    for a, b in (
        ("apps/research-lab", "saebase/research_lab"),
        ("docs", "harness/github_development_doc"),
        ("scripts", "saebase/tools"),
        ("examples", "saebase/examples"),
        ("assets", "saebase/assets"),
    ):
        move(a, b)
    plan = ROOT / "work_plan"
    dest = ROOT / "harness/github_work_plan"
    dest.mkdir(parents=True, exist_ok=True)
    for entry in plan.iterdir():
        move(f"work_plan/{entry.name}", f"harness/github_work_plan/{entry.name}")
    plan.rmdir()
    put("saebase/research_lab/__init__.py", '"""SAEBase Research Lab frontend and web API."""\n')
    put("harness/github_development_doc/README.md",
        "# SAEBase Developer Documentation\n\nDevelopment guides, API references, benchmark reports and release notes for SAEBase.\n\n"
        "[Getting Started](getting-started.md) · [Python API](python-api.md) · "
        "[Research Harness](../README.md) · [Release Notes](releases/v0.6.0.md)\n")
    for entry in (ROOT / "harness/github_development_doc").rglob("*.md"):
        new_src = entry.relative_to(ROOT).as_posix()
        old_src = new_src.replace("harness/github_development_doc/", "docs/", 1)
        entry.write_text(fix_markdown_links(entry.read_text(encoding="utf-8"), old_src, new_src),
                         encoding="utf-8")
    for entry in dest.rglob("*.md"):
        new_src = entry.relative_to(ROOT).as_posix()
        if not new_src.startswith("harness/github_work_plan/SAEBASE-"):
            old_src = new_src.replace("harness/github_work_plan/", "work_plan/", 1)
            entry.write_text(fix_markdown_links(entry.read_text(encoding="utf-8"), old_src, new_src),
                             encoding="utf-8")

def merge_datasets() -> None:
    DATA.mkdir(exist_ok=True)
    # Existing SAEBase readers are the reference implementations for public APIs.
    # The relocated OpenResearch artifacts, manifest and prompt readers remain
    # in datasets/ alongside them, without two implementations of each reader.
    for src in sorted(OLD_DATA.rglob("*.py")):
        relative = src.relative_to(OLD_DATA)
        dest = DATA / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = src.read_text(encoding="utf-8")
        dest.write_text(old, encoding="utf-8")
        if relative.name == "__init__.py" and relative.parent == Path("."):
            continue
        if relative == Path("plugins.py"):
            continue
        module = ".".join(relative.with_suffix("").parts)
        package = module[:-9] if module.endswith(".__init__") else module
        alias = "saebase.datasets" + ("." + package if package else "")
        src.write_text(
            '"""Legacy import forwarding to the canonical Dataset implementation."""\n'
            'from importlib import import_module as _import_module\n'
            f'_impl = _import_module("{alias}")\n'
            'def __getattr__(name):\n    return getattr(_impl, name)\n'
            'def __dir__():\n    return sorted(set(globals()) | set(dir(_impl)))\n'
            'for _name in dir(_impl):\n'
            '    if not _name.startswith("_"):\n        globals()[_name] = getattr(_impl, _name)\n',
            encoding="utf-8",
        )
    delete("datasets/__init__.py")
    put("saebase/datasets/__init__.py",
        '"""Source-tree alias: physical Dataset implementations live in /datasets.\n'
        'Installed wheels package them under saebase.datasets to avoid shadowing Hugging Face.\n'
        '"""\nfrom pathlib import Path\n'
        '_root = Path(__file__).resolve().parents[2] / "datasets"\n'
        'if _root.is_dir() and str(_root) not in __path__:\n'
        '    __path__.append(str(_root))\n')
    # Retain original dataset identity and availability and the research-specific
    # registered training dataset plugin in one registration module.
    original = (OLD_DATA / "plugins.py").read_text(encoding="utf-8")
    research = (DATA / "plugins.py").read_text(encoding="utf-8")
    start = research.find("NLP_NAMES =")
    if start < 0:
        raise RuntimeError("Research Dataset registry shape changed")
    joined = original.replace(
        "from saebase.registry import register_prompt_source, register_prompt_source_alias",
        "from saebase.registry import (register_prompt_source, register_prompt_source_alias, register_dataset)"
    ) + "\n\n" + research[start:]
    put("datasets/plugins.py", joined)
    put("saebase/data/plugins.py",
        '"""Compatibility import for Dataset plugin registration."""\n'
        'from saebase.datasets.plugins import *\n')
    # Only one physical implementation file per dataset API. The external
    # Hugging Face datasets namespace remains available without sys.modules hacks.
    (DATA / "NLP").rmdir() if (DATA / "NLP").exists() and not list((DATA / "NLP").iterdir()) else None

def unify_registry() -> None:
    reg = (ROOT / "saebase/registry.py").read_text(encoding="utf-8")
    if "def register_dataset(" in reg:
        raise RuntimeError("Unified Dataset registry already exists")
    marker = "\n\n@dataclass(frozen=True)\nclass ModelPlugin:"
    registry_type = """
@dataclass(frozen=True)
class DatasetPlugin:
    setup: Callable[[Any, DictConfig, Any], None]
    assert_task_supported: Callable[[DictConfig], None]
    eval_split_note: Callable[[Any], str]

"""
    reg = reg.replace(marker, "\n\n" + registry_type + marker)
    reg = reg.replace("_PROMPT_SOURCE_REGISTRY: dict[str, PromptSourcePlugin] = {}", 
        "_DATASET_REGISTRY: dict[str, DatasetPlugin] = {}\n_DATASET_ALIASES: dict[str, str] = {}\n_PROMPT_SOURCE_REGISTRY: dict[str, PromptSourcePlugin] = {}")
    reg += """

def register_dataset(name: str, setup, assert_task_supported, eval_split_note) -> None:
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset already registered: {name}")
    _DATASET_REGISTRY[name] = DatasetPlugin(setup, assert_task_supported, eval_split_note)

def register_dataset_alias(alias: str, canonical: str) -> None:
    if alias in _DATASET_ALIASES:
        raise ValueError(f"Dataset alias already registered: {alias}")
    _DATASET_ALIASES[alias] = canonical

def resolve_dataset_name(name: str) -> str:
    return _DATASET_ALIASES.get(str(name), str(name))

def get_dataset_plugin(name: str) -> DatasetPlugin:
    _ensure_plugins()
    key = resolve_dataset_name(name)
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset.name: {name}")
    return _DATASET_REGISTRY[key]

def list_registered_dataset_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_DATASET_REGISTRY)
"""
    put("saebase/registry.py", reg)
    put("saebase/plugin_bootstrap.py",
        '"""Single SAEBase registry bootstrap for datasets, models and research tasks."""\n'
        'from __future__ import annotations\n'
        '_loaded = False\n'
        'def ensure_plugins_loaded() -> None:\n'
        '    global _loaded\n'
        '    if _loaded:\n        return\n'
        '    import saebase.datasets.plugins  # noqa: F401\n'
        '    import models.plugins  # noqa: F401\n'
        '    import tasks.plugins  # noqa: F401\n'
        '    _loaded = True\n')
    legacy_models = (ROOT / "saebase/model_plugins.py").read_text(encoding="utf-8")
    put("models/plugins.py", legacy_models)
    put("saebase/model_plugins.py",
        '"""Compatibility re-export of the unified model plugin module."""\n'
        'from models.plugins import *\n'
        'from models import plugins as _impl\n'
        'def __getattr__(name):\n    return getattr(_impl, name)\n')
    tasks_code = '''"""Register all shipped SAEBase research tasks in one Registry."""
from saebase.registry import register_task
from saebase.sae.task import SAETask
from tasks.capture.task import CaptureTask
from tasks.intervene.task import InterveneTask
from tasks.generate.task import GenerateTask
from tasks.circuits.task import CircuitsTask
from tasks.explain.task import ExplainTask
from tasks.benchmark.task import BenchmarkTask
from tasks.train.task import TrainTask

for key, cls in (
    ("sae", SAETask),
    ("capture", CaptureTask),
    ("intervene", InterveneTask),
    ("generate", GenerateTask),
    ("circuits", CircuitsTask),
    ("explain", ExplainTask),
    ("benchmark", BenchmarkTask),
    ("train", TrainTask),
):
    register_task(key, cls)
'''
    put("tasks/plugins.py", tasks_code)
    put("saebase/tasks/plugins.py",
        '"""Compatibility import of the unified Task registry."""\n'
        'from tasks.plugins import *\n')
    delete("tasks/dlbase_plugins.py")
    delete("models/dlbase_plugins.py")
    delete("tasks/runner.py")

def unify_model() -> None:
    old = (ROOT / "models/gemma2_2b_base/model_task_dataset_compatibility.py").read_text(encoding="utf-8")
    new = (ROOT / "models/gemma2_2b_base/openresearch_compatibility.py").read_text(encoding="utf-8")
    # SAE (original) and seven LLM research tasks use the same model, with
    # different task-input config schemes.
    old = old.replace("def assert_compatible(cfg: DictConfig) -> None:\n",
        "def assert_compatible(cfg: DictConfig) -> None:\n"
        "    if str(cfg.task.name) != 'sae':\n"
        "        from saebase.datasets.prompt_schema import assert_available\n"
        "        task = str(cfg.task.name)\n"
        "        if task not in {'capture','intervene','generate','circuits','explain','benchmark','train'}:\n"
        "            raise ValueError(f'Unsupported SAE task: {task}')\n"
        "        name = str(cfg.dataset.name)\n"
        "        assert_available(name, 'gemma2_2b_base')\n"
        "        return\n")
    put("models/gemma2_2b_base/model_task_dataset_compatibility.py", old)
    delete("models/gemma2_2b_base/openresearch_compatibility.py")
    delete("models/gemma2_2b_base/openresearch_adapter.py")

def new_runtime() -> None:
    init = ROOT / "saebase/runtime/__init__.py"
    old = init.read_text(encoding="utf-8")
    init.write_text(old + '''

def __getattr__(name: str):
    if name == "TaskModule":
        from saebase.registry import TaskModule
        return TaskModule
    if name == "apply_lightning_reproducibility":
        from saebase.runtime.training.seed import apply_lightning_reproducibility
        return apply_lightning_reproducibility
    if name == "verify_training_runtime":
        from saebase.runtime.runtime_check import verify_training_runtime
        return verify_training_runtime
    raise AttributeError(name)
''', encoding="utf-8")
    entry = ROOT / "saebase/runtime/train.py"
    code = entry.read_text(encoding="utf-8")
    code = code.replace("config_path=\\"../conf\\"", "config_path=\\"../../conf\\"")
    code = code.replace("config_name=\\"openresearch\\"", "config_name=\\"sae_research\\"")
    code = code.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]")
    code = code.replace("dlbase.plugin_bootstrap", "saebase.plugin_bootstrap")
    code = code.replace("tasks.runner", "saebase.runner")
    put("saebase/runtime/train.py", code)
    src = ROOT / "conf/openresearch.yaml"
    if src.exists():
        src.rename(ROOT / "conf/sae_research.yaml")

def plain_rewrite(s: str, *, test_file: bool=False) -> str:
    # Exact Python identifiers before human-readable filesystem paths.
    direct = [
        ("models.LLM.gemma2_2b_base.model_task_dataset_compatibility", "models.gemma2_2b_base.model_task_dataset_compatibility"),
        ("models.LLM.gemma2_2b_base.gemma2_2b_base", "models.gemma2_2b_base.gemma2_2b_base"),
        ("research_datasets.NLP.", "saebase.datasets."),
        ("research_datasets.", "saebase.datasets."),
        ("tasks.LLM.", "tasks."),
        ("dlbase.registry", "saebase.registry"),
        ("dlbase.plugin_bootstrap", "saebase.plugin_bootstrap"),
        ("dlbase.", "saebase.runtime."),
        ("from scripts ", "from saebase.tools "),
        ("import scripts.", "import saebase.tools."),
        ("from scripts.", "from saebase.tools."),
        ("scripts.", "saebase.tools."),
        ("from examples.", "from saebase.examples."),
        ("import examples.", "import saebase.examples."),
        ("examples.", "saebase.examples."),
    ]
    if not test_file:
        direct.append(("saebase.data.", "saebase.datasets."))
    for a, b in direct:
        s = s.replace(a, b)
    source = [
        ("models/LLM/gemma2_2b_base/model_task_dataset_compatibility.py",
         "models/gemma2_2b_base/model_task_dataset_compatibility.py"),
        ("models/LLM/gemma2_2b_base/gemma2_2b_base.py",
         "models/gemma2_2b_base/gemma2_2b_base.py"),
        ("research_datasets/NLP/", "datasets/"),
        ("research_datasets/", "datasets/"),
        ("tasks/LLM/", "tasks/"),
        ("apps/research-lab", "saebase/research_lab"),
        ("dlbase/", "saebase/runtime/"),
        ("docs/", "harness/github_development_doc/"),
        ("work_plan/", "harness/github_work_plan/"),
        ("scripts/", "saebase/tools/"),
        ("examples/", "saebase/examples/"),
        ("assets/", "saebase/assets/"),
    ]
    for a, b in source:
        s = s.replace(a, b)
    s = s.replace("conf/openresearch.yaml", "conf/sae_research.yaml")
    s = s.replace("python -m dlbase.train", "python -m saebase.runtime.train")
    s = s.replace("python -m saebase.runtime.train", "python -m saebase.runtime.train")
    # Avoid duplicate package prefixes induced by sequential textual replacements.
    s = s.replace("saebase.saebase.runtime.", "saebase.runtime.")
    s = s.replace("saebase.saebase.datasets.", "saebase.datasets.")
    return s

def rewrite_tree() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIX and path.name not in {"Dockerfile","MANIFEST.in"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(s) for s in SKIP_PREFIX) or ".git" in path.parts:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        changed = plain_rewrite(original, test_file=rel.startswith("tests/"))
        if changed != original:
            path.write_text(changed, encoding="utf-8")
    # Remove stale GitHub Actions references to directories that no longer exist.
    for path in ROOT.glob(".github/workflows/*.yml"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("saebase/runtime/train.py", "saebase/runtime/train.py")
        path.write_text(text, encoding="utf-8")

def packaging() -> None:
    py = ROOT / "pyproject.toml"
    content = py.read_text(encoding="utf-8")
    content = content.replace('include = ["saebase*", "models*", "tasks*", "scripts*", "conf*", "dlbase*", "research_datasets*"]',
                              'include = ["saebase*", "models*", "tasks*", "conf*"]')
    content = content.replace('saebase = ["**/*.md", "**/*.yaml", "**/*.json"]',
                              'saebase = ["**/*.md", "**/*.yaml", "**/*.json", "**/*.html", "**/*.css", "**/*.js"]')
    py.write_text(content, encoding="utf-8")
    put("setup.py", '''"""Map physical /datasets sources into the saebase.datasets Python namespace.
This avoids shadowing the third-party Hugging Face datasets package.
"""
from pathlib import Path
from setuptools import find_packages, setup
datasets = Path(__file__).resolve().parent / "datasets"
packages = set(find_packages(exclude=("tests*",)))
packages.update({"saebase.datasets"})
for init in datasets.rglob("__init__.py"):
    parent = init.parent.relative_to(datasets)
    if parent.parts:
        packages.add("saebase.datasets." + ".".join(parent.parts))
setup(packages=sorted(packages), package_dir={"saebase.datasets": "datasets"})
''')
    manifest = ROOT / "MANIFEST.in"
    text = manifest.read_text(encoding="utf-8")
    text += "\nrecursive-include datasets *.py\nrecursive-include saebase/research_lab *.html *.css *.js *.json\n"
    manifest.write_text(text, encoding="utf-8")
    # Show the actual source path layout, not the old standalone DLBase package.
    for path in ("README.md", "README_cn.md"):
        f = ROOT / path
        c = f.read_text(encoding="utf-8")
        c = c.replace("python -m dlbase.train", "python -m saebase.runtime.train")
        f.write_text(c, encoding="utf-8")

def main() -> None:
    move_project_resources()
    merge_datasets()
    unify_registry()
    unify_model()
    new_runtime()
    rewrite_tree()
    packaging()
    # The one-time conversion should never ship as a runtime dependency.
    # Keep the script until all CI results have been inspected.
    print("Architecture migration applied; verify with CI before removing this temporary tool.")

if __name__ == "__main__":
    main()
