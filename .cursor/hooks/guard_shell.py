"""Conservative tool-level shell gate; only reviewed commands can mutate tasks.

Read-only inspection is permitted outside a plan. Do not treat this hook as a
security boundary against an agent with arbitrary filesystem/process access.
"""

from __future__ import annotations
import json
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GOVERNANCE = re.compile(r"(?i)(?:harness/core/|harness/visualization/|\.cursor/|\.github/)")
DANGEROUS = re.compile(
    r"(?i)(?:\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f|push\s+--force)"
    r"|\b(?:sudo|nohup|disown)\b|\brm\s+-[a-z]*r[a-z]*f\b"
    r"|\b(?:curl|wget)\b.{0,120}\|\s*(?:sh|bash))"
)
# A normal request such as "测 Harness H01" may run pytest with root cache
# creation disabled inside the
# dedicated Harness test sandbox without formal Agent Flow authorization.
# Other command families still require exact signed-off commands.
HARNESS_PYTEST = re.compile(
    r"^(?:python|python3) -m pytest -p no:cacheprovider (?:-q )?harness/tests/"
    r"[A-Za-z0-9_./:-]+(?: -q)?$"
)
SAFE_COMMANDS = [
    re.compile(r"^(?:git)\s+(?:status|diff|log|show|rev-parse)(?:\s+[^;&|<>]*)?$"),
    re.compile(r"^(?:ls|pwd|rg|grep|find)\b[^;&|<>]*$"),
    re.compile(
        r"^(?:python|python3)\s+-m\s+harness\.core\.agent_flow\s+"
        r"(?:intake|inspect|browse|preview|scope|stage|observe)(?:\s+[^;&|<>]*)?$"
    ),
]


def gate(command, plan_id=None):
    if isinstance(command, list):
        # argv form does not evaluate shell expansion.
        command = shlex.join([str(arg) for arg in command])
    if not isinstance(command, str) or not command.strip():
        return "Unknown shell command"
    command = command.strip()
    # Allow only mkdir -p inside the isolated Harness test directory.
    # Ordinary new project-root directories and production output paths
    # remain blocked, even when an Agent Flow lists a shell command.
    if re.match(r"(?i)^(?:mkdir\b|install\s+-d\b|git\s+clone\b)", command):
        try:
            args = shlex.split(command)
            if len(args) == 3 and args[:2] == ["mkdir", "-p"]:
                candidate = args[2]
                if not candidate.startswith("harness/tests/") or ".." in candidate.split("/"):
                    return "New project-root directories are forbidden"
                resolved = (ROOT / candidate).resolve()
                if not resolved.is_relative_to((ROOT / "harness/tests").resolve()):
                    return "Directory creation escapes the Harness test sandbox"
                acceptance = (ROOT / "harness/tests/wp18/acceptance").resolve()
                if resolved == acceptance or resolved.is_relative_to(acceptance):
                    return "Independent acceptance is read-only"
                return None
        except (ValueError, OSError):
            pass
        return "Shell directory creation is only allowed inside harness/tests/"
    if DANGEROUS.search(command):
        return "High-risk shell operation is not permitted to task agents"
    # Do not grant arbitrary shell execution just because a command mentions
    # harness/tests. Only the single-purpose, path-checked pytest form is free.
    if HARNESS_PYTEST.fullmatch(command):
        try:
            tokens = shlex.split(command)
            target = next(x for x in tokens if x.startswith("harness/tests/"))
            if ".." in target.split("/") or target.startswith("harness/tests//"):
                return "Test path traversal forbidden"
            if target.startswith("harness/tests/wp18/acceptance/"):
                return "Independent acceptance tests are not runnable by the task agent"
        except (ValueError, StopIteration):
            return "Malformed Harness test command"
        return None
    if GOVERNANCE.search(command) and "harness.core.agent_flow" not in command:
        return "Core/Harness governance sources are shielded from task shell access"
    if any(pattern.fullmatch(command) for pattern in SAFE_COMMANDS):
        # Generic recursion would read Core and the tool policies; require a
        # specifically named non-governance source tree for searches.
        try:
            args = shlex.split(command)
        except ValueError:
            return "Malformed inspection command"
        if args[0] in ("rg", "grep", "find"):
            permitted = (
                "tasks/",
                "models/",
                "datasets/",
                "saebase/",
                "conf/",
                "doc/knowledge/",
                "doc/code/",
            )
            paths = [
                value
                for value in args[1:]
                if any(value == root.rstrip("/") or value.startswith(root) for root in permitted)
            ]
            if not paths:
                return "Search must specify an explicit non-Core tree"
        if args[0] == "git" and len(args) > 1:
            if args[1] == "show" and "--stat" not in args:
                return "Full Git show can disclose Core; use --stat or scoped browser"
            if args[1] == "diff" and "--stat" not in args:
                path_args = args[args.index("--") + 1 :] if "--" in args else []
                if not path_args or not all(
                    any(
                        v == root.rstrip("/") or v.startswith(root)
                        for root in (
                            "tasks/",
                            "models/",
                            "datasets/",
                            "saebase/",
                            "conf/",
                            "doc/knowledge/",
                            "doc/code/",
                        )
                    )
                    for v in path_args
                ):
                    return "Diff must be --stat or restrict files to non-Core trees"
            if args[1] == "log" and any(x in args for x in ("-p", "--patch", "--stat")):
                return "Log inspection is summary-only; use scoped source browsing"
        return None
    ident = plan_id if plan_id is not None else os.getenv("OPENRESEARCH_ACTIVE_AGENT_FLOW")
    if not ident:
        return "Execution or mutation requires a human-authorized Agent Flow"
    try:
        from harness.core.agent_flow import load

        ledger = load(ident)
        if ledger["phase"] not in ("authorized", "partial", "review"):
            return "Current plan does not permit execution"
        if (
            not ledger["authorization"]
            or ledger["authorization"]["preview_sha256"] != ledger["preview_sha256"]
        ):
            return "Human-approved command scope missing"
        # An exact command match prevents wildcard approvals and shell expansion.
        if command not in ledger["authorization"]["commands"]:
            return "Command is outside the approved action/experiment list"
    except Exception as exc:
        return "Invalid active plan: " + str(exc)
    return None


def main():
    event = json.load(sys.stdin)
    command = event.get("command") or (event.get("tool_input") or {}).get("command")
    reason = gate(command)
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
        print(json.dumps({"permission": "deny", "agent_message": "Shell gate failed: " + str(exc)}))
