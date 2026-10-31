"""Preserve the historic v0.5.0 Beta evidence across newer releases.

This test does not mistake a successful package build for independent GPU validation.
"""

from pathlib import Path
import ast
import tomllib

ROOT = Path(__file__).resolve().parents[3]


def test_v050_package_version_and_release_notes():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "opensae-saebase"
    assert tuple(int(part) for part in project["version"].split(".")) >= (0, 6, 0)
    src = ast.parse((ROOT / "saebase" / "__init__.py").read_text())
    found = [
        node.value.value
        for node in src.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert found == [project["version"]]
    changelog = (ROOT / "harness/github/development_doc/CHANGELOG.md").read_text()
    assert "## 0.5.0" in changelog
    notes = (ROOT / "harness/github/development_doc/releases/v0.5.0.md").read_text()
    assert "Public Research Beta" in notes
    assert "16–32" in notes
    assert "Demo" in notes
    assert "not a stable production API" in notes
    assert "v0.5.0" in notes
    assert f"version: {project['version']}" in (ROOT / "CITATION.cff").read_text()
    assert "## 0.6.0" in changelog
    assert (ROOT / "harness/github/development_doc/releases/v0.6.0.md").is_file()
    assert (ROOT / f"harness/github/development_doc/releases/v{project['version']}.md").is_file()
    assert f"## {project['version']}" in changelog


def test_v050_distinguishes_open_playground_from_owner_gpu_compute():
    notes = (ROOT / "harness/github/development_doc/releases/v0.5.0.md").read_text()
    assert "operator authentication" in notes
    assert "no implied PyPI publication" in notes
    assert "no reliable" not in notes  # human-facing copy stays specific
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    for text in ("README.md", "README_cn.md"):
        assert f"harness/github/development_doc/releases/v{version}.md" in (ROOT / text).read_text()
    backend = (ROOT / "saebase/research_lab/backend/private_api.py").read_text()
    assert "OPENSAE_ENABLE_PRIVATE_JOBS" in backend
    assert "len(operator) < 32" in backend
