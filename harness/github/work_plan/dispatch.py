"""Run approved Work Plan command DAGs using existing SAEBase entrypoints.

This Harness coordinator neither defines scientific algorithms nor grants
network, GPU, native or publication permissions on the user's behalf.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
import time

from .lifecycle import ROOT, PlanError, check, source


def approved_command(step, root):
    """Use only repo-owned study scripts or known SAEBase module entrypoints."""
    argv = step.get("command")
    if (
        not isinstance(argv, list)
        or not 2 <= len(argv) <= 30
        or any(not isinstance(a, str) or not a or "\n" in a for a in argv)
    ):
        raise PlanError("Each executable step needs explicit argv")
    if argv[0] == "bash":
        if not argv[1].startswith("scripts/") or not argv[1].endswith(".sh"):
            raise PlanError("Only repository scripts may launch via bash")
        source(root, argv[1])
    elif argv[:2] == ["python", "-m"]:
        if len(argv) < 3 or not argv[2].startswith(("saebase.", "scripts.experiments.")):
            raise PlanError("Only SAEBase or study module entrypoints allowed")
        if "-c" in argv or "--eval" in argv:
            raise PlanError("Inline code is forbidden")
    else:
        raise PlanError("No arbitrary binary or shell expression allowed")
    if any(x in {"|", "&&", ";", "||", "`", "$("} for x in argv):
        raise PlanError("Shell syntax forbidden")
    permissions = set(step.get("permissions", [step.get("permission")]))
    requirements = {
        "--allow-local-gpu": "gpu",
        "--allow-native-compute": "native",
        "--approved": "publish",
    }
    for flag, permission in requirements.items():
        if flag in argv and permission not in permissions:
            raise PlanError("Execution flag lacks declared permission: " + flag)
    return list(argv)


def plan_commands(plan, root, allowed):
    """Identical declarative graph regardless of the user's prompt length."""
    result = []
    for step in plan["steps"]:
        argv = approved_command(step, root)
        permissions = step.get("permissions", [step.get("permission")])
        result.append(
            {
                "id": step["id"],
                "needs": step["needs"],
                "permissions": permissions,
                "on_failure": step.get("on_failure", "global_block"),
                "status": "PLANNED" if set(permissions) <= allowed else "NOT_AUTHORIZED",
                "argv": argv,
            }
        )
    return result


def execute(plan_path, run_id, allowed, *, root=ROOT):
    """Run observed commands; independent branches survive a local block."""
    plan = check(plan_path, root=root)
    if plan["status"] != "approved":
        raise PlanError("Unapproved Work Plan cannot run")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", run_id):
        raise PlanError("Invalid unique run ID")
    steps = plan_commands(plan, root, allowed)
    study = plan["experiment"]["study_id"]
    directory = root / "experiment" / "_studies" / study / "metadata" / run_id
    if directory.exists():
        raise PlanError("Existing run may not be overwritten")
    directory.mkdir(parents=True, exist_ok=False)
    ledger = {
        "schema": "saebase.workplan.dispatch.v1",
        "plan_id": plan["plan_id"],
        "source_plan": plan_path,
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    ledger["source_commit"] = head.stdout.strip() if head.returncode == 0 else None
    previous = {}
    global_block = False
    for step in steps:
        if (
            global_block
            or step["status"] != "PLANNED"
            or any(previous.get(d) != "PASS" for d in step["needs"])
        ):
            reason = (
                "GLOBAL_BLOCK"
                if global_block
                else "NOT_AUTHORIZED"
                if step["status"] != "PLANNED"
                else "DEPENDENCY_BLOCKED"
            )
            previous[step["id"]] = "BLOCKED"
            ledger["steps"].append({**step, "status": "BLOCKED", "reason": reason})
            continue
        argv = step["argv"]
        if argv[0] == "python":
            argv = [sys.executable, *argv[1:]]
        started = time.monotonic()
        try:
            with (
                (directory / (step["id"] + ".stdout.log")).open("x", encoding="utf-8") as out,
                (directory / (step["id"] + ".stderr.log")).open("x", encoding="utf-8") as err,
            ):
                code = subprocess.run(
                    argv, cwd=root, stdout=out, stderr=err, check=False, timeout=172800
                ).returncode
        except subprocess.TimeoutExpired:
            code = 124
        except OSError:
            code = 127
        state = "PASS" if code == 0 else "BLOCKED"
        previous[step["id"]] = state
        ledger["steps"].append(
            {
                **step,
                "status": state,
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": step["id"] + ".stdout.log",
                "stderr": step["id"] + ".stderr.log",
            }
        )
        if state != "PASS" and step["on_failure"] != "local_block":
            global_block = True
    ledger["status"] = "PASS" if all(s == "PASS" for s in previous.values()) else "PARTIAL"
    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
    with (directory / "dispatch.json").open("x", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    for name in ("network", "gpu", "native", "publish"):
        p.add_argument("--allow-" + name, action="store_true")
    args = p.parse_args()
    allowed = {"local"}
    for name in ("network", "gpu", "native", "publish"):
        if getattr(args, "allow_" + name):
            allowed.add(name)
    try:
        plan = check(args.plan)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "PREVIEW",
                        "steps": plan_commands(plan, ROOT, allowed),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        receipt = execute(args.plan, args.run_id, allowed)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if receipt["status"] == "PASS" else 3
    except (PlanError, OSError, ValueError, TypeError, KeyError) as exc:
        p.exit(2, "WORK PLAN BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
