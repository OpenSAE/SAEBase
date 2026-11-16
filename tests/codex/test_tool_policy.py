"""Protocol-level negatives for native Codex task policy."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("harness.core.tool_policy", reason="Optional central Hook integration")

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".codex/hooks"
sys.path.insert(0, str(HOOKS))
spec = importlib.util.spec_from_file_location("tool_policy", HOOKS / "tool_policy.py")
tool_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool_policy)


@pytest.fixture(autouse=True)
def regular_policy(monkeypatch):
    """Test regular policy independently of the human-selected active project mode."""
    from harness.core import tool_policy as central_policy

    monkeypatch.setattr(tool_policy, "current_mode", lambda: "regular")
    monkeypatch.setattr(central_policy, "read_mode", lambda root: "regular")


def test_intensive_policy_obeys_explicit_mode_without_changing_project_settings(monkeypatch):
    monkeypatch.setattr(tool_policy, "current_mode", lambda: "intensive")
    assert tool_policy.decision(event("Write", {"path": "tasks/example.py"})) is None
    assert tool_policy.decision(event("Write", {"path": "/workspace/ResearchHarness/other.py"}))


def event(name, data, cwd=ROOT):
    return {"hook_event_name": "PreToolUse", "tool_name": name, "tool_input": data, "cwd": str(cwd)}


def test_patch_all_paths_and_moves():
    assert tool_policy.patch_paths(
        "*** Begin Patch\n*** Update File: tasks/a.py\n*** Move to: tasks/b.py\n@@\n-x\n+y\n*** Delete File: tasks/c.py\n*** End Patch"
    ) == ["tasks/a.py", "tasks/b.py", "tasks/c.py"]


@pytest.mark.parametrize(
    "patch", ["", "*** Begin Patch\n*** End Patch", "*** Begin Patch\n*** Unknown\n*** End Patch"]
)
def test_bad_patch(patch):
    with pytest.raises(ValueError):
        tool_policy.patch_paths(patch)


def test_mixed_patch_is_denied():
    patch = "*** Begin Patch\n*** Add File: tests/codex_sample/a.py\n+x=1\n*** Add File: .researchharness/evil.py\n+x=2\n*** End Patch"
    assert tool_policy.decision(event("apply_patch", {"command": patch}))


def test_disposable_test_edit_and_subdirectory():
    patch = "*** Begin Patch\n*** Add File: tests/codex_sample/a.py\n+x=1\n*** End Patch"
    assert tool_policy.decision(event("apply_patch", {"command": patch})) is None
    patch = "*** Begin Patch\n*** Add File: codex_sample/a.py\n+x=1\n*** End Patch"
    assert tool_policy.decision(event("apply_patch", {"command": patch}, ROOT / "tests")) is None


@pytest.mark.parametrize(
    "name,data",
    [
        ("mcp__github__update_file", {"path": "tasks/a.py"}),
        ("unknown_tool", {}),
        ("Bash", {"command": 'python -c "print(1)"'}),
        ("Write", {"path": "tasks/AGENTS.md"}),
        ("Write", {"path": ".agents/skills/a/SKILL.md"}),
    ],
)
def test_unsupported_and_protected(name, data):
    assert tool_policy.decision(event(name, data))


def test_cli_error_is_a_native_deny():
    result = subprocess.run(
        [sys.executable, str(HOOKS / "tool_policy.py")],
        input="not-json",
        text=True,
        capture_output=True,
        check=True,
    )
    decision = json.loads(result.stdout)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
