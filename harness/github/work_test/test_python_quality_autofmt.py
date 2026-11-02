"""Confirm staged-only Ruff formatting, safety and CI parity without installing Ruff."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from saebase.tools import check_new_python_quality as quality


def test_staged_additions_only_and_no_frozen_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("added.py").write_text("x=1\n")
    Path("modified.py").write_text("y=2\n")
    frozen = Path("harness/tests/wp21/fixtures")
    frozen.mkdir(parents=True)
    (frozen / "locked.py").write_text("z=3\n")
    commands = []

    def run(args, **_):
        commands.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=b"added.py\0modified.py\0harness/tests/wp21/fixtures/locked.py\0",
        )

    monkeypatch.setattr(quality.subprocess, "run", run)
    assert quality.added_python_files() == ["added.py", "modified.py"]
    assert commands[0][:5] == [
        "git",
        "diff",
        "--name-only",
        "--diff-filter=A",
        "-z",
    ]
    assert quality.added_python_files(include_modified=True) == [
        "added.py",
        "modified.py",
    ]
    assert "--diff-filter=AM" in commands[1]


def test_fix_runs_format_before_read_only_checks(monkeypatch):
    commands = []
    monkeypatch.setattr(quality, "added_python_files", lambda *_a, **_k: ["new.py"])
    monkeypatch.setattr(quality, "verify_ruff_version", lambda: None)

    def run(args, **_):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(quality.subprocess, "run", run)
    assert quality.verify_new_files(fix=True) == 0
    assert commands == [
        ["ruff", "format", "new.py"],
        ["ruff", "check", "new.py"],
        ["ruff", "format", "--check", "new.py"],
    ]


def test_check_mode_never_writes(monkeypatch):
    commands = []
    monkeypatch.setattr(quality, "added_python_files", lambda *_a, **_k: ["new.py"])
    monkeypatch.setattr(quality, "verify_ruff_version", lambda: None)

    def run(args, **_):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(quality.subprocess, "run", run)
    assert quality.verify_new_files() == 0
    assert all(args != ["ruff", "format", "new.py"] for args in commands)


def test_fails_closed_on_version_drift(monkeypatch):
    def run(args, **_):
        return subprocess.CompletedProcess(args, 0, stdout="ruff 0.17.3\n")

    monkeypatch.setattr(quality.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="ruff==0.15.22"):
        quality.verify_ruff_version()


def test_no_files_needs_no_ruff_binary(monkeypatch):
    monkeypatch.setattr(quality, "added_python_files", lambda *_a, **_k: [])
    monkeypatch.setattr(
        quality,
        "verify_ruff_version",
        lambda: pytest.fail("Ruff must not run with empty selection"),
    )
    assert quality.verify_new_files() == 0


def test_precommit_uses_fix_and_ci_version():
    root = Path(__file__).resolve().parents[3]
    config = (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python saebase/tools/check_new_python_quality.py --fix" in config
    assert "ruff==0.15.22" in project
    assert "ruff==0.15.22" in ci
    assert "ruff format --check" not in config
