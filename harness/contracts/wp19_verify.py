"""WP19 staged, versioned, governance-owned executable examination gate.

Only H27-H46 are qualified. H47-H226 are visible proposals and MUST NOT
return a pass or accept a candidate until independently frozen and qualified.
Candidate execution is untrusted; use a restricted disposable reviewer host.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.contracts import registry, wp19_reference
from harness.contracts.selection import audit_selection

ROOT = registry.ROOT
PLAN = "harness/github_work_plan/19_harness_large_scale_examination_and_verifier_assurance.md"
SPEC = ROOT / "harness/contracts/wp19_cases.json"
RUNS = ROOT / "harness/tests/wp19/runs"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")
READY = {f"H{i}" for i in range(27, 47)}
ALL = {f"H{i}" for i in range(27, 227)}


def load_spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def check_spec() -> dict:
    data = load_spec()
    cases = data.get("cases")
    if (
        data.get("schema") != "saebase.wp19.case_catalog.v1"
        or not isinstance(cases, list)
        or [x.get("id") for x in cases] != [f"H{i}" for i in range(27, 227)]
    ):
        raise ValueError("WP19 case IDs/schema incomplete")
    ready = {x["id"] for x in cases if x.get("status") == "READY"}
    if ready != READY:
        raise ValueError("Only reviewed P0 cases can be marked READY")
    checks = 0
    for case in cases:
        if case["id"] not in ready:
            if case.get("status") != "PROPOSED" or "checks" in case:
                raise ValueError("Proposed case must not look qualified")
            continue
        case_id = case["id"]
        selected = registry.resolve(PLAN, case_id)
        if selected["tags"] != case["tags"]:
            raise ValueError("WP19 case tag mismatch")
        if set(selected["contract_ids"]) != set(
            registry.catalog()["required_always"] + ["CT-ERROR"] + case["contracts"]
        ):
            raise ValueError("WP19 case contract mismatch")
        scenarios = case.get("checks")
        if not isinstance(scenarios, list) or len(scenarios) < 4:
            raise ValueError("Insufficient checks")
        if (
            sum("expected" in s for s in scenarios) < 2
            or sum(s.get("reject") is True for s in scenarios) < 2
        ):
            raise ValueError("Need at least two positive and negative examples")
        for item in scenarios:
            if not isinstance(item.get("input"), dict):
                raise ValueError("Bad check input")
            if ("expected" in item) == (item.get("reject") is True):
                raise ValueError("Ambiguous expected outcome")
            try:
                observed = wp19_reference.solve(case_id, item["input"])
            except ValueError:
                if item.get("reject") is not True:
                    raise ValueError(
                        f"Reference rejected positive {case_id}/{item['name']}"
                    ) from None
            else:
                if item.get("reject") or observed != item["expected"]:
                    raise ValueError(
                        f"Reference disagrees with frozen spec {case_id}/{item['name']}"
                    )
            checks += 1
    return {
        "cases": len(cases),
        "ready": len(ready),
        "proposed": len(cases) - len(ready),
        "checks": checks,
    }


def verify_case(case_id: str, path: Path) -> dict:
    if case_id not in READY:
        raise ValueError("Case is only a proposal, not READY")
    if path.name != case_id or not RUN_ID.fullmatch(path.parent.name):
        raise ValueError("Case directory and run ID mismatch")
    path = path.resolve(strict=True)
    if not path.is_relative_to(RUNS.resolve()):
        raise ValueError("Candidate path escapes WP19 sandbox")
    source = path / "implementation.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("Missing or symlinked implementation.py")
    # A task solution cannot delegate its implementation to the official
    # reference oracle. Public reference may be read but not re-exported.
    try:
        nodes = ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "harness.contracts.wp19_reference"
            ):
                raise ValueError("Candidate imports governance reference oracle")
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("harness.contracts.wp19_reference")
                for alias in node.names
            ):
                raise ValueError("Candidate imports governance reference oracle")
    except (UnicodeError, SyntaxError) as exc:
        raise ValueError("Candidate source is invalid") from exc

    case = next(c for c in load_spec()["cases"] if c["id"] == case_id)
    outcomes = []
    for trial in case["checks"]:
        with tempfile.TemporaryDirectory(prefix="wp19-review-") as tmp:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": tmp,
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(ROOT / "harness/contracts/wp18_worker.py"),
                        str(source),
                    ],
                    input=json.dumps(trial["input"], allow_nan=False),
                    text=True,
                    capture_output=True,
                    cwd=tmp,
                    env=env,
                    timeout=6,
                    check=False,
                )
                result = json.loads(proc.stdout)
                if proc.returncode != 0 or not isinstance(result, dict):
                    passed = False
                elif trial.get("reject") is True:
                    passed = (
                        result.get("ok") is False
                        and result.get("kind") == "ValueError"
                        and result.get("mutated") is False
                    )
                else:
                    passed = (
                        result.get("ok") is True
                        and result.get("mutated") is False
                        and result.get("output") == trial["expected"]
                    )
            except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
                passed = False
            outcomes.append({"name": trial["name"], "passed": bool(passed)})
    return {
        "case": case_id,
        "passed": sum(x["passed"] for x in outcomes),
        "failed": sum(not x["passed"] for x in outcomes),
        "checks": outcomes,
        "human_approval": False,
    }


def audit_scope(base_sha: str, head_sha: str = "HEAD") -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{40}", base_sha):
        raise ValueError("Exact base commit SHA required")
    if head_sha != "HEAD" and not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("Exact candidate SHA required")
    proc = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", base_sha, head_sha],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    bad = []
    for row in proc.stdout.splitlines():
        status, path = row.split("\t", 1)
        parts = Path(path).parts
        allowed = (
            status == "A"
            and len(parts) >= 7
            and parts[:4] == ("harness", "tests", "wp19", "runs")
            and bool(RUN_ID.fullmatch(parts[4]))
            and parts[5] in READY
        )
        if not allowed:
            bad.append(path)
    return bad


def scan() -> list[dict]:
    if not RUNS.exists():
        return []
    reports = []
    for run in sorted(RUNS.iterdir()):
        if not run.is_dir() or not RUN_ID.fullmatch(run.name):
            continue
        existing = {d.name for d in run.iterdir() if d.is_dir()}
        if not existing:
            continue
        for case_id in sorted(READY, key=lambda x: int(x[1:])):
            target = run / case_id
            if not target.is_dir():
                reports.append({"case": case_id, "failed": 1, "missing": True})
            else:
                reports.append(verify_case(case_id, target))
        for unknown in sorted(existing - READY):
            reports.append({"case": unknown, "failed": 1, "not_ready": True})
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-spec", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--head-sha")
    args = parser.parse_args()
    try:
        result: dict = {"spec": check_spec()}
        if args.head_sha and not args.base:
            raise ValueError("--head-sha requires --base")
        if args.base:
            result["unauthorized_changes"] = audit_scope(args.base, args.head_sha or "HEAD")
        if args.case:
            if args.run_dir is None:
                raise ValueError("Required --run-dir")
            result["runs"] = [verify_case(args.case, args.run_dir)]
        elif args.scan:
            result["runs"] = scan()
        elif not args.base and not args.check_spec:
            raise ValueError("Select --check-spec or a case or --scan")
        fail = bool(result.get("unauthorized_changes")) or any(
            x["failed"] for x in result.get("runs", [])
        )
        result["status"] = "FAIL" if fail else "PASS"
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
