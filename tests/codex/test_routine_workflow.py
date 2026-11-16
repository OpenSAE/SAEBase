"""Exercise actual Hook decisions, including the handoff commands that failed."""

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

pytest.importorskip("harness.core.tool_policy", reason="Optional central Hook integration")

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".codex/hooks"
sys.path.insert(0, str(HOOKS))
spec = importlib.util.spec_from_file_location("routine_tool_policy", HOOKS / "tool_policy.py")
tool_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool_policy)


def regular_policy(monkeypatch):
    """Exercise regular guards without changing the active human development choice."""
    from harness.core import tool_policy as central_policy

    monkeypatch.setattr(tool_policy, "current_mode", lambda: "regular")
    monkeypatch.setattr(central_policy, "read_mode", lambda root: "regular")


@pytest.mark.parametrize(
    "command",
    [
        "id",
        "cat /workspace/ResearchHarness/migration/SAEBASE_HANDOFF.md",
        "cat .agents/skills/researchharness/SKILL.md",
        "python -m pytest -q tests/codex",
        "python -m ruff check tests/codex",
        "ls -la .codex",
        "rg -n hooks .codex",
        "rg --files --hidden .codex",
        "find .codex -type f",
        "git diff -- .codex",
        "python -m py_compile .codex/hooks/tool_policy.py",
    ],
)
def test_previously_blocked_commands_now_reach_execution(command):
    event = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(ROOT),
        "tool_input": {"command": command},
    }
    result = subprocess.run(
        [sys.executable, str(HOOKS / "tool_policy.py")],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {}


@pytest.mark.parametrize(
    "path",
    [
        "tasks/example/operation.py",
        "tests/example/test_operation.py",
        "doc/code/code_annotations.json",
        "README.md",
    ],
)
def test_normal_source_edit_needs_no_research_flow(path):
    assert (
        tool_policy.decision(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "cwd": str(ROOT),
                "tool_input": {"path": path},
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "path",
    [
        ".agents/skills/example/SKILL.md",
        ".researchharness/project.yaml",
        "atlas/core.html",
        "harness/tests/wp18/acceptance/result.json",
        "/workspace/ResearchHarness/src/harness/cli.py",
    ],
)
def test_protected_writes_still_fail(path, monkeypatch):
    regular_policy(monkeypatch)
    assert tool_policy.decision(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "cwd": str(ROOT),
            "tool_input": {"path": path},
        }
    )


@pytest.mark.parametrize(
    "path",
    [
        ".codex/hooks.json",
        ".codex/config.toml",
        ".codex/hooks/tool_policy.py",
        ".codex/rules/example.rules",
        ".codex/AGENTS.md",
    ],
)
def test_active_development_policy_allows_codex(path):
    assert (
        tool_policy.decision(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "cwd": str(ROOT),
                "tool_input": {"path": path},
            }
        )
        is None
    )


def test_native_permissions_cover_entire_codex():
    import tomllib

    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    assert config["permissions"]["saebase"]["filesystem"][":workspace_roots"][".codex"] == "write"


def test_web_and_connectors_are_outside_local_hook_matcher():
    groups = json.loads((ROOT / ".codex/hooks.json").read_text())["hooks"]["PreToolUse"]
    for name in ["web.run", "browser", "mcp__search_service__search", "mcp__github__get_repo"]:
        assert not any(re.search(group["matcher"], name) for group in groups)
    for name in ["Bash", "Write", "apply_patch"]:
        assert any(re.search(group["matcher"], name) for group in groups)


def test_manual_diagnostic_reports_client_failure(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("session_diagnostic", HOOKS / "preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    monkeypatch.setattr(preflight, "current_mode", lambda: "regular")
    monkeypatch.setattr(preflight, "client_state", lambda: ({}, {}))
    monkeypatch.setattr(preflight, "inspect_state", lambda *args: ["Synthetic sandbox unavailable"])
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "codex-cli 0.166.0\n", ""),
    )
    assert preflight.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["errors"] == ["Synthetic sandbox unavailable"]


def test_manual_diagnostic_checks_the_registered_authorization_hook(monkeypatch):
    import tomllib

    spec = importlib.util.spec_from_file_location("hook_diagnostic", HOOKS / "preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    monkeypatch.setattr(preflight, "current_mode", lambda: "regular")
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    config["default_permissions"] = "saebase"
    hook = {
        "sourcePath": str(ROOT / ".codex/hooks.json"),
        "eventName": "preToolUse",
        "enabled": True,
        "trustStatus": "trusted",
    }
    hooks = {"data": [{"hooks": [hook]}]}
    assert preflight.inspect_state({"config": config}, hooks) == []
    hook["trustStatus"] = "untrusted"
    assert preflight.inspect_state({"config": config}, hooks)
