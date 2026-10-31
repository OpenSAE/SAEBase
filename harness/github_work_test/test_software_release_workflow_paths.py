"""Keep the generic software release workflow aligned with current repository paths."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = (
    "release_candidate.py",
    "verify_distribution.py",
    "check_installed_wheel.py",
    "check_sdist_roundtrip.py",
)


def test_software_release_workflow_uses_existing_maintenance_tools() -> None:
    workflow = (ROOT / ".github/workflows/software-release.yml").read_text(
        encoding="utf-8"
    )
    for name in TOOLS:
        path = ROOT / "saebase/tools" / name
        assert path.is_file(), f"missing maintained tool: {path}"
        assert f"python saebase/tools/{name}" in workflow
    assert "from saebase.tools.release_candidate import project_info" in workflow
    assert "python scripts/" not in workflow
    assert "from scripts.release_candidate" not in workflow
    assert '--notes-file "harness/github_development_doc/releases/$TAG.md"' in workflow
    assert 'PUBLISH-v$VERSION' in workflow
    assert "SHA256SUMS" in workflow


def test_release_metadata_tracks_project_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]
    package_init = (ROOT / "saebase/__init__.py").read_text(encoding="utf-8")
    changelog = (ROOT / "harness/github_development_doc/CHANGELOG.md").read_text(
        encoding="utf-8"
    )
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    release_notes = (
        ROOT / "harness/github_development_doc/releases" / f"v{version}.md"
    )
    assert re.search(rf'^__version__ = "{re.escape(version)}"$', package_init, re.M)
    assert re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog, re.M)
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M)
    assert release_notes.is_file() and release_notes.read_text(encoding="utf-8").strip()
