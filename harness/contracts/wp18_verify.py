"""Frozen, read-only behavioral verification for WP18 H07-H26.

Only reviewer CI should execute untrusted task code. Hooks are not OS sandboxes.
The task agent may READ all cases and verifier sources but MUST NOT edit them.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.contracts import registry
from harness.contracts.selection import audit_selection

ROOT = registry.ROOT
RUNS = ROOT / "harness/tests/wp18/runs"
SPEC = ROOT / "harness/contracts/wp18_cases.json"
PLAN = "harness/github_work_plan/18_harness_local_agent_examination_and_independent_acceptance.md"
REQUIRED = {f"H{i:02d}" for i in range(7, 27)}
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")


def load_spec() -> dict:
    record = json.loads(SPEC.read_text(encoding="utf-8"))
    if record.get("schema") != "saebase.wp18.behavior_cases.v1":
        raise ValueError("Wrong frozen WP18 verification schema")
    return record


def validate_spec() -> dict:
    document = load_spec()
    cases = document.get("cases")
    if not isinstance(cases, list) or {x["id"] for x in cases} != REQUIRED:
        raise ValueError("Exactly twenty unique H07-H26 cases are required")
    count = 0
    for case in cases:
        case_id = case["id"]
        selected = registry.resolve(PLAN, case_id)
        if set(selected["contract_ids"]) != set(
            registry.catalog()["required_always"] + ["CT-ERROR"] + case["contracts"]
        ):
            raise ValueError(f"{case_id} selected contracts differ from frozen Work Plan")
        if selected["tags"] != case["tags"]:
            raise ValueError(f"{case_id} task tag mismatch")
        if audit_selection(selected["contract_ids"], case["signals"]):
            raise ValueError(f"{case_id} missing IO contract")
        if not case.get("goal") or not case.get("rule"):
            raise ValueError(f"{case_id} must define behavior and normative rule")
        scenarios = case["checks"]
        names = [row["name"] for row in scenarios]
        if len(set(names)) != len(names):
            raise ValueError(f"{case_id} duplicate scenario name")
        if (
            sum("expected" in x for x in scenarios) < 2
            or sum(x.get("reject") is True for x in scenarios) < 2
        ):
            raise ValueError(f"{case_id} requires positive and negative cases")
        for row in scenarios:
            if not isinstance(row.get("input"), dict):
                raise ValueError(f"{case_id} non-object input")
            if ("expected" in row) == (row.get("reject") is True):
                raise ValueError(f"{case_id} scenario must expect an output or rejection")
            if "expected" in row and not isinstance(row["expected"], dict):
                raise ValueError(f"{case_id} non-object expected output")
        count += len(scenarios)
    return {"cases": len(cases), "checks": count, "contracts": len(registry.catalog()["contracts"])}


def run_case(case_id: str, run_dir: Path) -> dict:
    if case_id not in REQUIRED:
        raise ValueError("Only H07-H26 are supported by this verification suite")
    if not RUN_ID.fullmatch(run_dir.parent.name):
        raise ValueError("Invalid run identifier")
    root = RUNS.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_relative_to(root) or run_dir.name != case_id:
        raise ValueError("Verification target outside selected WP18 case directory")
    source = run_dir / "implementation.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("Missing or symlinked candidate implementation.py")
    case = next(x for x in load_spec()["cases"] if x["id"] == case_id)
    checks = []
    for row in case["checks"]:
        with tempfile.TemporaryDirectory(prefix="saebase-wp18-") as temp:
            env = {
                "HOME": temp,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT),
                "PATH": os.environ.get("PATH", ""),
            }
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(ROOT / "harness/contracts/wp18_worker.py"),
                        str(source),
                    ],
                    input=json.dumps(copy.deepcopy(row["input"]), allow_nan=False),
                    capture_output=True,
                    text=True,
                    cwd=temp,
                    env=env,
                    timeout=8,
                    check=False,
                )
                output = json.loads(proc.stdout)
            except (subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
                checks.append(
                    {"name": row["name"], "passed": False, "reason": "timeout_or_invalid_output"}
                )
                continue
            if proc.returncode != 0 or not isinstance(output, dict):
                passed = False
            elif row.get("reject") is True:
                passed = (
                    output.get("ok") is False
                    and output.get("kind") == "ValueError"
                    and output.get("mutated") is False
                )
            else:
                actual = output.get("output")
                expected = row["expected"]
                try:
                    passed = (
                        output.get("ok") is True
                        and output.get("mutated") is False
                        and json.dumps(actual, sort_keys=True, allow_nan=False)
                        == json.dumps(expected, sort_keys=True, allow_nan=False)
                    )
                except (TypeError, ValueError):
                    passed = False
            checks.append({"name": row["name"], "passed": passed})
    return {
        "case": case_id,
        "checks": checks,
        "passed": sum(x["passed"] for x in checks),
        "failed": sum(not x["passed"] for x in checks),
        "technical_only": True,
        "human_approved": False,
    }


def audit_changed_scope(base: str) -> list[str]:
    """Reject every WP18 task-agent change outside a NEW case sandbox.

    Enforced by trusted CI on pull requests with a wp18/ branch. No grading
    implementation, spec, rules, or historical run can be changed by a task PR.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Base must be an exact commit SHA")
    diff = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", base, "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    bad = []
    for line in diff.stdout.splitlines():
        status, path = line.split("\t", 1)
        parts = Path(path).parts
        valid = (
            status == "A"
            and len(parts) >= 7
            and parts[:4] == ("harness", "tests", "wp18", "runs")
            and bool(RUN_ID.fullmatch(parts[4]))
            and re.fullmatch(r"H(?:0[1-9]|1[0-9]|2[0-6])", parts[5]) is not None
        )
        if not valid:
            bad.append(path)
    return bad


def scan() -> list[dict]:
    if not RUNS.is_dir():
        return []
    results = []
    for run in sorted(RUNS.iterdir()):
        if not run.is_dir() or not RUN_ID.fullmatch(run.name):
            continue
        for case in sorted(REQUIRED):
            target = run / case
            if target.is_dir():
                results.append(run_case(case, target))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-spec", action="store_true")
    parser.add_argument("--case", choices=sorted(REQUIRED))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--base", help="Exact PR base SHA for a task-agent scope audit")
    args = parser.parse_args()
    try:
        result = {"spec": validate_spec()}
        if args.base:
            result["unauthorized_changes"] = audit_changed_scope(args.base)
        if args.case:
            if not args.run_dir:
                raise ValueError("--case requires --run-dir")
            result["runs"] = [run_case(args.case, args.run_dir)]
        elif args.scan:
            result["runs"] = scan()
        elif not args.check_spec and not args.base:
            raise ValueError("Choose --check-spec, --case, or --scan")
        failed = bool(result.get("unauthorized_changes")) or any(
            item["failed"] for item in result.get("runs", [])
        )
        result["status"] = "FAIL" if failed else "PASS"
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
