"""Regression checks for the narrowly delegated WP18 Harness test sandbox.

These checks belong to independent governance, not writable exam submissions.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_hook(name: str):
    path = ROOT / ".cursor" / "hooks" / (name + ".py")
    spec = importlib.util.spec_from_file_location("wp18_" + name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_root(tmp_path: Path, monkeypatch):
    for name in ("harness", "saebase", ".cursor", ".github", "datasets", "models", "doc"):
        (tmp_path / name).mkdir()
    hooks = load_hook("guard_paths")
    monkeypatch.setattr(hooks, "ROOT", tmp_path)
    return hooks


def test_agent_may_edit_disposable_harness_test_files_without_flow(tmp_path, monkeypatch):
    hooks = prepare_root(tmp_path, monkeypatch)
    candidates = (
        "harness/tests/wp18/runs/example/H01/implementation.py",
        "harness/tests/wp18/runs/example/H01/test_implementation.py",
        "harness/tests/wp18/runs/example/H01/fixtures/sample.json",
        "harness/tests/wp18/runs/example/H01/review_candidate.json",
        "harness/tests/other_topic/run42/report.md",
    )
    for path in candidates:
        assert hooks.gate("create_file", {"file_path": path}, active=None) is None, path


@pytest.mark.parametrize(
    "path",
    [
        "harness/core/agent_flow.py",
        "harness/code/build.py",
        "harness/visualization/app.js",
        ".cursor/rules/core/00-core.mdc",
        ".cursor/hooks/guard_paths.py",
        ".github/workflows/ci.yml",
        "saebase/runtime/train.py",
        "harness/tests/README.md",
        "harness/tests/wp18/README.md",
        "harness/tests/wp18/acceptance/example/H01/verdict.md",
    ],
)
def test_governance_and_independent_review_are_denied(tmp_path, monkeypatch, path):
    hooks = prepare_root(tmp_path, monkeypatch)
    assert hooks.gate("write_file", {"file_path": path}, active=None) is not None


@pytest.mark.parametrize(
    "path", ["utils/helper.py", "outputs/report.json", "scripts/test.py", "custom_tests/unit.py"]
)
def test_unknown_new_root_directory_is_denied_even_when_named(tmp_path, monkeypatch, path):
    hooks = prepare_root(tmp_path, monkeypatch)
    assert "top-level" in hooks.gate("write_file", {"file_path": path}, active=None)


def test_only_approved_root_exception_names_need_formal_flow(tmp_path, monkeypatch):
    hooks = prepare_root(tmp_path, monkeypatch)
    for path in ("experiments/research_run/metrics.json", "tests/unassigned_test.py"):
        assert hooks.gate("write_file", {"file_path": path}, active=None) is not None
    from harness.core import agent_flow

    monkeypatch.setattr(agent_flow, "load", lambda _: {"phase": "authorized"})
    monkeypatch.setattr(agent_flow, "scope", lambda _obj, _files: None)
    for path in ("experiments/research_run/metrics.json", "tests/unassigned_test.py"):
        assert hooks.gate("write_file", {"file_path": path}, active="TRUSTED") is None


def test_root_files_and_mixed_scope_are_denied(tmp_path, monkeypatch):
    hooks = prepare_root(tmp_path, monkeypatch)
    assert hooks.gate("write_file", {"file_path": "new_root_file.py"}, active=None)
    assert hooks.gate(
        "write_file",
        {
            "files": [
                {"file_path": "harness/tests/wp18/runs/x/H01/implementation.py"},
                {"file_path": "harness/core/agent_flow.py"},
            ]
        },
        active=None,
    )


def test_symlink_escape_and_parent_traversal_are_denied(tmp_path, monkeypatch):
    hooks = prepare_root(tmp_path, monkeypatch)
    external = tmp_path.parent / (tmp_path.name + "_outside")
    external.mkdir()
    (tmp_path / "harness" / "tests").mkdir()
    (tmp_path / "harness" / "tests" / "escape").symlink_to(external, target_is_directory=True)
    assert hooks.gate("write_file", {"file_path": "harness/tests/escape/file.py"}, active=None)
    assert hooks.gate(
        "write_file", {"file_path": "harness/tests/../core/agent_flow.py"}, active=None
    )


def test_shell_only_runs_scoped_pytest_not_governance_or_arbitrary_scripts():
    hooks = load_hook("guard_shell")
    assert (
        hooks.gate(
            "python -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/demo/H01/test_implementation.py"
        )
        is None
    )
    assert (
        hooks.gate("python3 -m pytest -p no:cacheprovider harness/tests/other_topic/test_sample.py")
        is None
    )
    assert hooks.gate("mkdir -p harness/tests/wp18/runs/example/H01") is None
    for command in (
        "python -m pytest -p no:cacheprovider -q harness/tests/../core/test_secrets.py",
        "python -m pytest -p no:cacheprovider -q harness/tests/wp18/acceptance/demo/H01/test_secret.py",
        "python -m pytest -q harness/github_work_test/test_atlas_semantic_audit.py",
        "mkdir -p novel_root",
        "mkdir -p harness/tests/../../novel_root",
        "mkdir -p harness/tests/wp18/acceptance/private",
        "mkdir -p harness/tests/wp18/acceptance",
        "mkdir -p harness/tests/./wp18/acceptance",
        "mkdir -p harness/tests/example && mkdir novel_root",
        "git clone https://example.com/project other_root",
        "python -c 'print(1)'",
    ):
        assert hooks.gate(command) is not None, command


def test_work_order_harness_reference_uses_real_repository_path(tmp_path, monkeypatch):
    path = ROOT / ".cursor/skills/workflow/run-work-order/scripts/validate_work_order.py"
    spec = importlib.util.spec_from_file_location("wp18_work_order", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "harness/github_work_plan").mkdir(parents=True)
    correct = (
        "harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md"
    )
    assert module.reference(correct, "harness", check_exists=False) == correct
    with pytest.raises(module.InvalidOrder):
        module.reference(
            "harness/github_harness/github_work_plan/18.md", "harness", check_exists=False
        )


def test_tracked_repository_has_only_approved_top_level_directories():
    # This checks committed code, in addition to the interactive edit gate.
    approved = {
        ".cursor",
        ".github",
        "conf",
        "datasets",
        "doc",
        "harness",
        "models",
        "saebase",
        "tasks",
        "experiments",
        "tests",
    }
    tree = subprocess.run(
        ["git", "ls-tree", "-d", "--name-only", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = set(tree.stdout.splitlines())
    assert observed <= approved, "Unexpected project root directory: " + repr(
        sorted(observed - approved)
    )
