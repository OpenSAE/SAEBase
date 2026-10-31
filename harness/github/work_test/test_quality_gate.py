"""Offline tests for gradual Ruff admission of new Python modules."""

from pathlib import Path
from subprocess import CompletedProcess

from saebase.tools import check_new_python_quality as quality


def test_new_file_discovery_uses_pr_base_and_excludes_existing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("new.py").write_text("x = 1\n", encoding="utf-8")
    Path("old.py").write_text("x = 2\n", encoding="utf-8")
    Path("notes.md").write_text("document\n", encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, stdout=b"new.py\0notes.md\0removed.py\0")

    monkeypatch.setattr(quality.subprocess, "run", fake_run)
    assert quality.added_python_files("BASE") == ["new.py"]
    assert calls[0][-2:] == ["BASE...HEAD", "--"]


def test_ruff_checks_new_files_without_fixing_them(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("added.py").write_text("value = 1\n", encoding="utf-8")
    commands = []

    def fake_added(_base_sha=None):
        return ["added.py"]

    def fake_run(args, **kwargs):
        commands.append(args)
        return CompletedProcess(args, 0)

    monkeypatch.setattr(quality, "added_python_files", fake_added)
    monkeypatch.setattr(quality.subprocess, "run", fake_run)
    assert quality.verify_new_files("BASE") == 0
    assert commands == [
        ["ruff", "check", "added.py"],
        ["ruff", "format", "--check", "added.py"],
    ]


def test_nonzero_ruff_result_fails_the_gate(monkeypatch):
    monkeypatch.setattr(
        quality,
        "added_python_files",
        lambda _base_sha=None: ["added.py"],
    )
    monkeypatch.setattr(
        quality.subprocess,
        "run",
        lambda args, **kwargs: CompletedProcess(args, 1),
    )
    assert quality.verify_new_files() == 1
