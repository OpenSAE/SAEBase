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



def test_api_stdin_formats_then_lints_without_file_write(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(quality, "verify_ruff_version", lambda: None)

    def run(args, **kwargs):
        calls.append((args, kwargs["input"]))
        if args[1] == "format" and "--check" not in args:
            return subprocess.CompletedProcess(args, 0, stdout="x = 1\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(quality.subprocess, "run", run)
    assert quality.format_api_source("x=1\n", "tasks/precheck/example.py") == "x = 1\n"
    assert calls == [
        (["ruff", "format", "--stdin-filename", "tasks/precheck/example.py", "-"], "x=1\n"),
        (["ruff", "check", "--stdin-filename", "tasks/precheck/example.py", "-"], "x = 1\n"),
        (["ruff", "format", "--check", "--stdin-filename", "tasks/precheck/example.py", "-"],
         "x = 1\n"),
    ]
    assert not (tmp_path / "tasks").exists()


def test_api_stdin_lint_failure_blocks_even_after_formatting(monkeypatch):
    monkeypatch.setattr(quality, "verify_ruff_version", lambda: None)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "check":
            return subprocess.CompletedProcess(args, 1, stdout="F821 undefined name", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="print(bad)\n", stderr="")

    monkeypatch.setattr(quality.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Ruff verification failed"):
        quality.format_api_source("print(bad)\n", "tasks/check.py")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "filename",
    [
        "../other.py", "/tmp/temp.py", "tasks/../unsafe.py", "tasks//example.py",
        "tasks\\windows.py", "tasks/../anything.py",
        "harness/tests/wp21/fixtures/frozen.py", "not_python.txt", "-bad.py",
    ],
)
def test_api_stdin_rejects_unsafe_filename_without_running_ruff(monkeypatch, filename):
    monkeypatch.setattr(
        quality, "verify_ruff_version",
        lambda: pytest.fail("Ruff must not run for invalid paths"),
    )
    with pytest.raises(ValueError, match="safe, unfrozen"):
        quality.format_api_source("x=1\n", filename)


def test_api_stdin_rejects_oversized_text_before_ruff(monkeypatch):
    monkeypatch.setattr(
        quality, "verify_ruff_version",
        lambda: pytest.fail("Ruff must not run for oversized source"),
    )
    with pytest.raises(ValueError, match="2 MB"):
        quality.format_api_source("x" * (quality.MAX_API_SOURCE_CHARS + 1), "tasks/test.py")


def test_api_stdin_cli_emits_only_final_clean_code(monkeypatch, capsys):
    import io
    import sys

    monkeypatch.setattr(sys, "argv", ["quality", "--stdin", "--filename", "tasks/sample.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("x=1\n"))
    monkeypatch.setattr(
        quality, "format_api_source",
        lambda source, filename: "x = 1\n" if filename == "tasks/sample.py" else "",
    )
    assert quality.main() == 0
    captured = capsys.readouterr()
    assert captured.out == "x = 1\n"
    assert captured.err == ""


def test_api_stdin_cli_never_emits_partial_source_on_failure(monkeypatch, capsys):
    import io
    import sys

    monkeypatch.setattr(sys, "argv", ["quality", "--stdin", "--filename", "tasks/sample.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("bad("))

    def refuse(*_args):
        raise RuntimeError("Ruff refused invalid code")

    monkeypatch.setattr(quality, "format_api_source", refuse)
    with pytest.raises(SystemExit) as caught:
        quality.main()
    assert caught.value.code == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "blocked" in out.err


def test_real_pinned_ruff_stdin_format_when_available():
    import shutil

    if shutil.which("ruff") is None:
        pytest.skip("Ruff executable unavailable outside CI")
    try:
        quality.verify_ruff_version()
    except RuntimeError:
        pytest.skip("Pinned Ruff version unavailable in this environment")
    assert quality.format_api_source("x=1\n", "tasks/precheck/example.py") == "x = 1\n"
