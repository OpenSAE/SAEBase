"""Bounded local WP21 assignment profile, installed by the trusted dispatcher.

This is a tool hook, not an OS sandbox or a human-review signature. A task
agent must not create/change the protected assignment record. There is no
automatic authorization when the package is unpacked or merely prepared.
"""

from __future__ import annotations
import hashlib
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment():
    path = ROOT / ".cursor/wp21_assignment.json"
    if path.is_symlink():
        raise ValueError("WP21 assignment must be a regular dispatcher-owned file")
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (
        obj.get("schema") != "saebase.wp21.assignment.v1"
        or obj.get("state") != "assigned"
        or not obj.get("assignment_ref")
    ):
        raise ValueError("WP21 is prepared but has no current user case assignment")
    for field, relative in [
        ("spec_sha256", "harness/tests/wp21/cases.json"),
        ("catalog_sha256", "harness/contracts/catalog.json"),
        ("protocol_sha256", "harness/contracts/REAL_WORKFLOW_PROTOCOL.md"),
    ]:
        if obj.get(field) != digest(ROOT / relative):
            raise ValueError("WP21 frozen specification changed")
    spec = json.loads((ROOT / "harness/tests/wp21/cases.json").read_text(encoding="utf-8"))
    case = next((x for x in spec["cases"] if x["id"] == obj.get("case_id")), None)
    if (
        not case
        or case["allowed_paths"] != obj.get("allowed_paths")
        or case["budget"] != obj.get("budget")
    ):
        raise ValueError("WP21 assignment differs from frozen case")
    blob = json.dumps(
        case, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    if hashlib.sha256(blob.encode()).hexdigest() != obj.get("case_sha256"):
        raise ValueError("WP21 case hash mismatch")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,47}", obj.get("run_id", "")) is None:
        raise ValueError("WP21 invalid run id")
    expected = f"harness/tests/wp21/runs/{obj['run_id']}/{obj['case_id']}"
    if obj.get("evidence_root") != expected:
        raise ValueError("WP21 evidence path mismatch")
    return obj


def allows_path(relative: str, obj: dict) -> bool:
    import fnmatch

    if (
        not isinstance(relative, str)
        or any(x in {"", ".", ".."} for x in relative.split("/"))
        or any(c in relative for c in "\\:\x00")
    ):
        return False
    path = ROOT
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            return False
    if not path.resolve().is_relative_to(ROOT.resolve()):
        return False
    evidence = obj["evidence_root"]
    if relative == evidence or relative.startswith(evidence + "/"):
        return True
    if relative.startswith("experiment/"):
        return True  # this entire precreated root belongs to this isolated case
    if (
        relative.startswith(("harness/", ".cursor/", ".github/", "doc/", "saebase/runtime/"))
        or "/" not in relative
    ):
        return False
    return any(fnmatch.fnmatchcase(relative, x) for x in obj["allowed_paths"])


def command_decision(command: str, _depth: int = 0):
    """Return (handled, rejection_or_None), without shell execution."""
    try:
        if _depth > 1:
            return True, "Nested command recorders are not an execution entrypoint"
        obj = assignment()
        if obj is None:
            return False, None
        if any(
            token in command for token in (";", "&&", "||", "|", ">", "<", "`", "$(", "\n", "\r")
        ):
            return True, "WP21 commands use one argv invocation, without shell composition"
        args = shlex.split(command)
        if not args:
            return True, "Empty WP21 command"
        if args[:2] == ["mkdir", "-p"] and len(args) == 3:
            target = Path(args[2])
            rel = (
                target.resolve().relative_to(ROOT.resolve()).as_posix()
                if target.is_absolute()
                else args[2]
            )
            return True, None if allows_path(
                rel + "/wp21_directory_probe", obj
            ) else "WP21 directory outside assigned case"
        executable = Path(args[0]).name
        if not re.fullmatch(r"python(?:3(?:\.(?:11|12))?)?", executable):
            return False, None  # existing read-only and dangerous-command gates still apply
        if any(x in args for x in ("-c", "-i", "--interactive")):
            return True, "Inline/arbitrary interpreter commands are outside the WP21 driver profile"
        if args[1:3] == ["-m", "pytest"]:
            prefix = f"tests/wp21/{obj['case_id']}"
            targets, cache_disabled = [], False
            i = 3
            while i < len(args):
                arg = args[i]
                if arg in {"-q", "-v", "-vv", "-x", "--disable-warnings"}:
                    i += 1
                    continue
                if (
                    arg == "-p"
                    and i + 1 < len(args)
                    and args[i + 1] == "no:cacheprovider"
                    and not cache_disabled
                ):
                    cache_disabled = True
                    i += 2
                    continue
                file_part = arg.split("::", 1)[0]
                if (
                    file_part == prefix or file_part.startswith(prefix + "/")
                ) and ".." not in file_part.split("/"):
                    targets.append(file_part)
                    i += 1
                    continue
                return (
                    True,
                    "Pytest options/targets must stay inside the assigned case; extra plugins/configs are forbidden",
                )
            if targets and cache_disabled:
                return True, None
            return True, "Use case-local pytest targets with -p no:cacheprovider"
        if args[1:3] == ["-m", "saebase.runtime.train"]:
            for arg in args[3:]:
                key, sep, value = arg.lstrip("+").partition("=")
                if arg.split("=", 1)[0] in {
                    "--config-path",
                    "--config-dir",
                    "--config-name",
                    "-cp",
                    "-cd",
                    "-cn",
                }:
                    return (
                        True,
                        "WP21 uses the frozen runtime config entry; external config roots are forbidden",
                    )
                if key in {"paths", "hydra", "experiment"} and sep:
                    return True, "Whole output/control mapping overrides are outside this profile"
                if sep and key.startswith("hydra."):
                    return True, "Changing Hydra output/control paths is outside this profile"
                if key == "paths.root" and Path(value).resolve() != ROOT.resolve():
                    return True, "Runtime output must use the assigned workspace"
                if key == "experiment.overwrite" and value.lower() != "false":
                    return True, "WP21 uses fresh attempt leaves, not overwrite"
            return True, None
        if len(args) >= 2:
            driver = args[1]
            prefix = f"tests/wp21/{obj['case_id']}/"
            if driver.startswith(prefix) and driver.endswith(".py") and allows_path(driver, obj):
                return True, None
            allowed_tools = {
                "harness/tests/wp21/tools/preflight.py",
                "harness/tests/wp21/tools/record.py",
            }
            if driver in allowed_tools:
                if "--freeze-assets" in args:
                    return True, "Shared checkpoint freezing belongs to the dispatcher"
                if "--output" in args:
                    value = args[args.index("--output") + 1]
                    target = Path(value)
                    rel = (
                        target.resolve().relative_to(ROOT.resolve()).as_posix()
                        if target.is_absolute()
                        else value
                    )
                    if not rel.startswith(obj["evidence_root"] + "/") or not allows_path(rel, obj):
                        return (
                            True,
                            "Command records must remain in the assigned evidence directory",
                        )
                if driver.endswith("record.py"):
                    if (
                        "--cwd" not in args
                        or Path(args[args.index("--cwd") + 1]).resolve() != ROOT.resolve()
                    ):
                        return True, "Recorded commands must run in the assigned source workspace"
                    if "--" not in args:
                        return True, "Recorder must specify its actual command argv"
                    nested = shlex.join(args[args.index("--") + 1 :])
                    handled, reason = command_decision(nested, _depth + 1)
                    if not handled or reason:
                        return True, reason or "Recorder command outside assigned profile"
                return True, None
        return False, None
    except (ValueError, KeyError, OSError, IndexError, TypeError) as exc:
        return True, "WP21 scope check failed: " + str(exc)
