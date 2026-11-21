"""Keep the generic software release workflow aligned with current repository paths."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("release.py", "distribution.py")


def test_software_release_workflow_uses_existing_maintenance_tools() -> None:
    workflow = (ROOT / ".github/workflows/software-release.yml").read_text(encoding="utf-8")
    for name in TOOLS:
        path = ROOT / "scripts" / name
        assert path.is_file(), f"missing maintained tool: {path}"
        assert f"python scripts/{name}" in workflow
    assert "from scripts.release import project_info" in workflow
    assert "from scripts.release_candidate" not in workflow
    assert '--notes-file "doc/code/documentation/source/development/releases/$TAG.md"' in workflow
    assert "PUBLISH-v$VERSION" in workflow
    assert "SHA256SUMS" in workflow


def test_release_metadata_tracks_project_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    package_init = (ROOT / "base/application/__init__.py").read_text(encoding="utf-8")
    changelog = (ROOT / "doc/code/documentation/source/development/CHANGELOG.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    release_notes = ROOT / "doc/code/documentation/source/development/releases" / f"v{version}.md"
    assert re.search(rf'^__version__ = "{re.escape(version)}"$', package_init, re.M)
    assert re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog, re.M)
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M)
    assert release_notes.is_file() and release_notes.read_text(encoding="utf-8").strip()
