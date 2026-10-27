"""Release-contract tests that require only the local checkout and core runtime."""

from __future__ import annotations

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_package_metadata_matches_public_api_version():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    from saebase import __version__

    assert data["project"]["version"] == __version__
    assert data["project"]["name"] == "opensae-saebase"
    assert data["project"]["license"] == "Apache-2.0"
    assert data["project"]["scripts"]["saebase"] == "saebase.cli:main"


def test_public_distribution_has_required_files():
    for rel in (
        "LICENSE",
        "NOTICE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        ".github/workflows/ci.yml",
        "MANIFEST.in",
        "harness/github_development_doc/PROVENANCE.md",
        "harness/github_development_doc/RELEASE_CHECKLIST.md",
        "conf/__init__.py",
        "conf/config.yaml",
    ):
        assert (ROOT / rel).is_file(), rel


def test_cli_uses_caller_working_directory():
    from main import parse_args, build_overrides

    caller = ROOT / "outside-example"
    args, _ = parse_args(["--model", "pythia70m", "--device", "cpu"])
    overrides = build_overrides(args, caller)
    assert f"paths.root={caller.resolve()}" in overrides
    assert not any(s.startswith("paths.root=") and "site-packages" in s for s in overrides)


def test_metadata_keeps_optional_backends_separate():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert all(
        "vllm" not in p.lower() and "nnsight" not in p.lower() for p in project["dependencies"]
    )
    assert any("vllm" in p for p in project["optional-dependencies"]["generation"])
    assert len(list((ROOT / "conf").rglob("*.yaml"))) >= 100
