"""Version-independent release gate checks with synthetic distributions."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from saebase.tools import release_candidate as gate

SHA = "a" * 40


def fake_project(tmp_path: Path, version: str = "0.3.0") -> tuple[Path, Path]:
    root = tmp_path / "root"
    dist = tmp_path / "dist"
    root.mkdir(parents=True)
    dist.mkdir(parents=True)
    (root / "saebase").mkdir()
    (root / "harness" / "github_development_doc" / "releases").mkdir(parents=True)
    entrypoints = {
        "saebase": "saebase.cli:main",
        "saebase-prepare-prompts": "saebase.cli:prepare_prompts",
        "saebase-check-env": "saebase.cli:check_env",
        "saebase-intervene": "saebase.cli:intervene",
        "saebase-benchmark": "saebase.cli:benchmark",
        "saebase-audit": "saebase.cli:audit",
        "saebase-ground-truth": "saebase.cli:ground_truth",
        "saebase-release-matrix": "saebase.cli:release_matrix",
        "saebase-release-gate": "saebase.cli:release_gate",
    }
    project = (
        '[project]\nname = "opensae-saebase"\n'
        f'version = "{version}"\n'
        "[project.scripts]\n"
        + "".join(f'{name} = "{command}"\n' for name, command in entrypoints.items())
    )
    (root / "pyproject.toml").write_text(project, encoding="utf-8")
    (root / "saebase" / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (root / "harness/github_development_doc/CHANGELOG.md").write_text(
        f"# Changelog\n\n## {version} - Candidate\n", encoding="utf-8"
    )
    (root / "harness" / "github_development_doc" / "releases" / f"v{version}.md").write_text(
        "# Candidate\n\nRelease notes.\n", encoding="utf-8"
    )
    (root / "LICENSE").write_text("Apache License, Version 2.0\n", encoding="utf-8")

    wheel = dist / f"opensae_saebase-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr(
            f"opensae_saebase-{version}.dist-info/METADATA",
            f"Name: opensae-saebase\nVersion: {version}\nLicense-Expression: Apache-2.0\n",
        )
        z.writestr(
            f"opensae_saebase-{version}.dist-info/entry_points.txt",
            "[console_scripts]\n" + "".join(f"{n} = {p}\n" for n, p in entrypoints.items()),
        )
        z.writestr("saebase/cli.py", "def main():\n    return 0\n")
        z.writestr("conf/config.yaml", "defaults: []\n")
    sdist = dist / f"opensae_saebase-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as t:
        for name in (
            "pyproject.toml",
            "saebase/__init__.py",
            "LICENSE",
            "harness/github_development_doc/CHANGELOG.md",
            f"harness/github_development_doc/releases/v{version}.md",
        ):
            payload = (root / name).read_bytes()
            entry = tarfile.TarInfo(f"opensae_saebase-{version}/{name}")
            entry.size = len(payload)
            t.addfile(entry, io.BytesIO(payload))
    return root, dist


def test_release_candidate_roundtrip_and_tamper_detection(tmp_path, monkeypatch):
    root, dist = fake_project(tmp_path)
    monkeypatch.setattr(gate, "source_sha", lambda _root: SHA)
    out = tmp_path / "candidate"
    evidence = gate.prepare(root, dist, out, SHA, expected_version="0.3.0")
    assert evidence["tag"] == "v0.3.0"
    assert evidence["status"] == "software_distribution_verified"
    assert "no new real checkpoint" in evidence["scientific_scope"]
    assert len(gate.verify_assets(dist, out, SHA, "0.3.0")["distribution_artifacts"]) == 2
    with (dist / "opensae_saebase-0.3.0.tar.gz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gate.verify_assets(dist, out, SHA, "0.3.0")


def test_version_must_match_package_source_and_notes(tmp_path, monkeypatch):
    root, dist = fake_project(tmp_path)
    (root / "saebase" / "__init__.py").write_text('__version__ = "0.4.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        gate.project_info(root)
    (root / "saebase" / "__init__.py").write_text('__version__ = "0.3.0"\n', encoding="utf-8")
    (root / "harness" / "github_development_doc" / "releases" / "v0.3.0.md").unlink()
    with pytest.raises(ValueError, match="notes missing"):
        gate.project_info(root)


def test_source_commit_must_be_exact_and_match_checkout(tmp_path, monkeypatch):
    root, dist = fake_project(tmp_path)
    monkeypatch.setattr(gate, "source_sha", lambda _root: SHA)
    with pytest.raises(ValueError, match="source commit"):
        gate.prepare(root, dist, tmp_path / "out", "f" * 40)
    with pytest.raises(ValueError, match="source commit"):
        gate.prepare(root, dist, tmp_path / "out", "not-a-commit")


def test_ci_jobs_require_all_six_successful_for_one_commit():
    jobs = [
        {
            "name": name,
            "head_sha": SHA,
            "status": "completed",
            "conclusion": "success",
        }
        for name in gate.REQUIRED_CI
    ]
    assert len(gate.verify_ci_jobs({"jobs": jobs}, SHA)) == 6
    jobs[-1]["conclusion"] = "skipped"
    with pytest.raises(ValueError, match="not successful"):
        gate.verify_ci_jobs({"jobs": jobs}, SHA)
    jobs[-1]["conclusion"] = "success"
    jobs.pop()
    with pytest.raises(ValueError, match="missing required"):
        gate.verify_ci_jobs({"jobs": jobs}, SHA)


def test_publish_rejects_old_and_existing_tags(tmp_path, monkeypatch):
    root, _ = fake_project(tmp_path)
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda *args, **kw: "v0.2.0\nv0.2.1\nstage-d-evidence-20261016\n",
    )
    assert gate.publish_eligible(root, "0.3.0")["newer_than_existing_software_tags"]
    old_root, _ = fake_project(tmp_path / "old", "0.2.1")
    with pytest.raises(ValueError, match="newer"):
        gate.publish_eligible(old_root, "0.2.1")


def test_release_workflow_is_manual_explicit_and_privileged_only_when_publish():
    path = gate.ROOT / ".github" / "workflows" / "software-release.yml"
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"verify", "publish"}
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["actions"] == "read"
    assert jobs["publish"]["permissions"]["contents"] == "write"
    assert jobs["publish"]["if"].count("publish") >= 1
    text = path.read_text(encoding="utf-8")
    assert "PUBLISH-v$VERSION" in text
    assert "actions/workflows/ci.yml/runs?" in text
    assert "git ls-remote --tags" in text
    assert "release-evidence.json" in text
    assert 'assets[name]["digest"] == "sha256:" + proof["sha256"]' in text


def test_current_repository_version_and_release_notes_are_coherent():
    import tomllib

    version, tag, entrypoints = gate.project_info()
    declared = tomllib.loads((gate.ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert version == declared
    assert tag == f"v{declared}"
    assert len(entrypoints) == 9
