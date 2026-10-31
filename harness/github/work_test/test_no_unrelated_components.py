"""SAEBase accepts LLM/SAE research components without unrelated Vision tasks."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
RETIRED_PATHS = (
    "datasets/vision",
    "models/vision",
    "tasks/vision",
    "conf/dataset/cub.yaml",
    "conf/model/resnet18_mlp.yaml",
    "conf/task/classification.yaml",
    "saebase/runtime/training/facets.py",
)
CONTENT_ROOTS = ("harness/core", "harness/code", "harness/knowledge", "harness/visualization", "doc", "datasets", "models", "tasks", ".cursor")
CONTENT_FILES = (
    "models/plugins.py",
    "tasks/plugins.py",
    "conf/sae_research.yaml",
    "conf/tasks_log.md",
    "README.md",
    "README_cn.md",
    "harness/github/development_doc/openresearch-integration.md",
)
RETIRED_TERMS = re.compile(
    r"(?i)CUB[-_ ]?200|ResNet(?:[-_ ]?18)?|"
    r"(?<![a-z])classification(?![a-z])|"
    r"(?<![a-z])vision(?![a-z])|鸟类|细粒度分类|视觉训练|视觉模型|图像分类"
)
TEXT_SUFFIXES = {".py", ".json", ".md", ".yaml", ".yml", ".mdc", ".js", ".html"}


def test_unrelated_components_absent():
    for relative in RETIRED_PATHS:
        assert not (ROOT / relative).exists(), relative


def test_saebase_and_llm_entrypoints_remain():
    assert (ROOT / "saebase/api/features.py").is_file()
    assert (ROOT / "saebase/data/prompt_store.py").is_file()
    for task in ("capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"):
        assert (ROOT / "tasks" / task / "task.py").is_file(), task


def test_no_retired_terms_in_integrated_sources():
    files = [ROOT / path for path in CONTENT_FILES]
    for root in CONTENT_ROOTS:
        files.extend(path for path in (ROOT / root).rglob("*") if path.suffix in TEXT_SUFFIXES)
    failures = []
    for path in files:
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RETIRED_TERMS.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    assert not failures, "\n".join(failures[:100])
