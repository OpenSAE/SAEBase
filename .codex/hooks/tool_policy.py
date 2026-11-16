"""目的和主要功能：按项目模式处理客户端工具请求。

适用范围：密集开发暂停中央审批，常规模式调用既有任务授权检查。
输入只作为请求资料解析，实际操作使用客户端权限与进程账号。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from development_mode import current_mode

ROOT = Path(__file__).resolve().parents[2]


def patch_paths(command: str) -> list[str]:
    """Extract every affected path; reject incomplete or ambiguous envelopes."""
    if not isinstance(command, str):
        raise ValueError("Patch command must be text")
    lines = command.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("Expected a complete apply_patch envelope")
    paths = []
    for line in lines[1:-1]:
        if line.startswith(("*** Add File: ", "*** Update File: ", "*** Delete File: ")):
            paths.append(line.split(": ", 1)[1])
        elif line.startswith("*** Move to: "):
            if not paths:
                raise ValueError("Move without a source path")
            paths.append(line.removeprefix("*** Move to: "))
        elif line.startswith("*** ") and line != "*** End of File":
            raise ValueError("Unknown patch directive")
    if not paths or any(not path or path != path.strip() for path in paths):
        raise ValueError("Patch has no auditable paths")
    return paths


def event_paths(event: dict) -> list[str]:
    data = event.get("tool_input")
    if not isinstance(data, dict):
        raise ValueError("Tool input must be an object")
    if event.get("tool_name") == "apply_patch":
        paths = patch_paths(data.get("command"))
    else:
        # Event decoding is local and works without importing the central engine.
        fields = {
            "file_path",
            "path",
            "relative_path",
            "target_file",
            "destination",
            "new_path",
            "old_path",
            "filename",
            "file",
        }

        def candidates(value):
            if isinstance(value, dict):
                for key, part in value.items():
                    if key in fields and isinstance(part, str):
                        yield part
                    elif isinstance(part, (dict, list)):
                        yield from candidates(part)
            elif isinstance(value, list):
                for part in value:
                    yield from candidates(part)

        paths = list(candidates(data))
    cwd = Path(event.get("cwd", ROOT)).resolve()
    if not cwd.is_relative_to(ROOT.resolve()):
        raise ValueError("Tool cwd is outside this repository")
    # Paths are relative to tool cwd, not necessarily the Git root.
    result = []
    for raw in paths:
        path = Path(raw)
        if ".." in path.parts or any(c in raw for c in ("\\", "\x00")):
            raise ValueError("Noncanonical tool path")
        result.append(str(path if path.is_absolute() else cwd / path))
    return result


def decision(event: dict) -> str | None:
    """None permits a known call; a string rejects it before side effects."""
    if not isinstance(event, dict) or event.get("hook_event_name") != "PreToolUse":
        return "Invalid PreToolUse event"
    name, data = event.get("tool_name"), event.get("tool_input")
    if not isinstance(data, dict):
        return "Missing structured tool input"
    if current_mode() == "intensive":
        # The user's persistent development choice suspends central approval.
        # Native client permissions and the process account remain in effect.
        if name in {"apply_patch", "Edit", "Write", "write_file", "create_file", "update_file"}:
            if any(
                not Path(path).resolve().is_relative_to(ROOT.resolve())
                for path in event_paths(event)
            ):
                return "Mutation escapes the project root"
        return None
    from harness.core.tool_policy import path_gate, shell_gate

    if name == "Bash":
        if Path(event.get("cwd", ROOT)).resolve() != ROOT.resolve():
            return "Run approved shell commands with workdir set to the Git root"
        return shell_gate(ROOT, data.get("command"))
    if name in {"apply_patch", "Edit", "Write", "write_file", "create_file", "update_file"}:
        paths = event_paths(event)
        if not paths:
            return "Mutation has no auditable paths"
        return path_gate(ROOT, "write_file", {"files": [{"path": p} for p in paths]})
    if name in {"read_file", "list_directory", "view_image"}:
        paths = event_paths(event)
        if not paths:
            return "Read requires explicit paths"
        from harness.core.intake import read_path

        for path in paths:
            reason = read_path(ROOT, path)
            if reason:
                return reason
        return None
    # Only the local tool events registered in hooks.json reach this entry.
    return "Unsupported tool: no reviewed adapter for " + str(name)


def main() -> int:
    try:
        reason = decision(json.load(sys.stdin))
    except Exception as exc:
        reason = "Tool authorization failed: " + str(exc)
    if reason:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
        )
    else:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
