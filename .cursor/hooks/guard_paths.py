"""Cursor edit gate: deny by default without a human-authorized Agent Flow.

This is a best-effort tool hook, not an OS sandbox. Governance is immutable
to the task agent even when a task includes its paths.
"""

from __future__ import annotations
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = json.loads((ROOT / ".cursor/policy.json").read_text(encoding="utf-8"))
PATH_FIELDS = {
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
EDIT_WORDS = ("write", "edit", "delete", "patch", "move", "rename", "create", "replace")
# Test-only writes are restricted to this sandbox and never imply permission
# to modify the Harness governance implementation or real research records.
HARNESS_TEST_SANDBOX = "harness/tests/"
NEW_TOP_LEVEL_DIRS = {"experiments", "tests"}


def candidate_paths(value):
    if isinstance(value, dict):
        for key, part in value.items():
            if key in PATH_FIELDS and isinstance(part, str):
                yield part
            elif isinstance(part, (dict, list)):
                yield from candidate_paths(part)
    elif isinstance(value, list):
        for part in value:
            yield from candidate_paths(part)


def gate(tool, payload, active=None):
    """Return rejection reason or None; never accept an edit with unknown scope."""
    import os

    if not any(word in str(tool).lower() for word in EDIT_WORDS):
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return "Unparseable mutation input"
    if not isinstance(payload, dict):
        return "Mutation tool input must be a mapping"
    paths = list(candidate_paths(payload))
    if not paths:
        return "Mutation tool supplies no auditable file paths"
    plan_id = active if active is not None else os.environ.get("OPENRESEARCH_ACTIVE_AGENT_FLOW")
    try:
        from harness.core.agent_flow import load, scope

        plan = None
        for path in paths:
            candidate = pathlib.Path(path)
            resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
            if not resolved.is_relative_to(ROOT):
                return "Agent path escapes repository root"
            if ".." in candidate.parts:
                return "Parent traversal is forbidden in task edit paths"
            rel = resolved.relative_to(ROOT).as_posix()
            parts = resolved.relative_to(ROOT).parts
            if len(parts) < 2:
                return "Task agents cannot create or edit root-level entries"
            if parts[0] not in NEW_TOP_LEVEL_DIRS and not (ROOT / parts[0]).is_dir():
                return "New top-level directories are forbidden except experiments/ and tests/"
            if rel.startswith(HARNESS_TEST_SANDBOX):
                # A test exercise may create code, fixtures, reports and
                # candidate trees here without exposing any governance files.
                continue
            if rel.startswith((".cursor/", "harness/", ".github/")):
                return "Harness/Core and execution rules are never editable by task agents"
            if rel == "saebase/runtime" or rel.startswith("saebase/runtime/"):
                return "SAEBase Runtime requires separate architecture governance"
            if parts[0] not in POLICY["approved_roots"] and parts[0] not in NEW_TOP_LEVEL_DIRS:
                return "No unscoped root entries are editable"
            if not plan_id:
                return "Non-test edits require a human-authorized Agent Flow"
            if plan is None:
                plan = load(plan_id)
            scope(plan, [rel])
    except Exception as exc:
        return "Failed human-approved path authorization: " + str(exc)
    return None


def main():
    event = json.load(sys.stdin)
    reason = gate(
        event.get("tool_name") or event.get("tool"),
        event.get("tool_input") or event.get("arguments") or {},
    )
    if reason:
        print(
            json.dumps(
                {"permission": "deny", "agent_message": reason, "user_message": reason},
                ensure_ascii=False,
            )
        )
    else:
        print("{}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"permission": "deny", "agent_message": "Path gate failed: " + str(exc)}))
