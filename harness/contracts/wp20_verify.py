"""WP20 independent behavior grader. Use only from trusted governance revision.

Candidate code is UNTRUSTED. Running it in a subprocess is NOT OS isolation;
the reviewer must invoke --case/--scan inside a restricted disposable sandbox.
The CI --check-spec mode never imports or executes submitted candidate code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from harness.contracts import registry
from harness.contracts import wp20_reference_a, wp20_reference_b, wp20_reference_c, wp20_reference_d

ROOT = registry.ROOT
RUNS = ROOT / "harness/tests/wp20/runs"
PLAN = "harness/github_work_plan/20_integrated_multistage_harness_exams.md"
SCENARIOS = ROOT / "harness/contracts/wp20_scenarios.json"
SPEC = ROOT / "harness/contracts/wp20_cases.json"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")
CASES = {
    **wp20_reference_a.CASES,
    **wp20_reference_b.CASES,
    **wp20_reference_c.CASES,
    **wp20_reference_d.CASES,
}
READY = {f"X{i:02d}" for i in range(1, 21)}
if set(CASES) != READY:
    raise ValueError("Missing independent WP20 reference")


def _tree(root):
    """Only materialized case artifacts, never the untrusted .git internals."""
    base = Path(root)
    rows = {}
    for f in base.rglob("*"):
        if f.is_symlink():
            raise ValueError("Symlink in candidate workdir")
        if f.is_file() and ".git" not in f.relative_to(base).parts:
            rows[f.relative_to(base).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return dict(sorted(rows.items()))


def _reference(case, payload):
    with tempfile.TemporaryDirectory(prefix="wp20-ref-") as wd:
        try:
            result = CASES[case](payload, Path(wd))
        except ValueError:
            return {"ok": False, "kind": "ValueError", "output": None, "files": _tree(wd)}
        return {"ok": True, "kind": None, "output": result, "files": _tree(wd)}


def load_scenarios():
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))


def check_spec():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    fixtures = load_scenarios()
    if spec.get("status") != "READY" or [c.get("id") for c in spec["cases"]] != sorted(READY):
        raise ValueError("WP20 cases are not frozen READY")
    if (
        fixtures.get("schema") != "saebase.wp20.behavioral_scenarios.v1"
        or set(fixtures["cases"]) != READY
    ):
        raise ValueError("WP20 behavioral fixtures do not cover all cases")
    checked = 0
    for case in sorted(READY):
        selected = registry.resolve(PLAN, case)
        entry = next(row for row in spec["cases"] if row["id"] == case)
        if entry["status"] != "READY" or entry["oracle_status"] != "QUALIFIED":
            raise ValueError(f"Unqualified {case}")
        if selected["tags"] != entry["tags"] or not set(entry["contracts"]) <= set(
            selected["contract_ids"]
        ):
            raise ValueError(f"Missing contract for {case}")
        tests = fixtures["cases"][case]
        if (
            len(tests) < 4
            or sum(row["valid"] is True for row in tests) < 2
            or sum(row["valid"] is False for row in tests) < 2
        ):
            raise ValueError(f"Incomplete positive and negative scenarios in {case}")
        if len({row["id"] for row in tests}) != len(tests):
            raise ValueError("Duplicate scenario")
        for row in tests:
            observed = _reference(case, row["input"])
            if observed["ok"] is not row["valid"]:
                raise ValueError(f"Reference outcome mismatch {case}/{row['id']}")
            if row["valid"] and not isinstance(observed["output"], dict):
                raise ValueError(f"Reference returned invalid output {case}/{row['id']}")
            checked += 1
    return {
        "cases": len(READY),
        "scenarios": checked,
        "status": "VERIFIER_QUALIFIED",
        "human_approval": False,
    }


def _candidate_source(candidate):
    source = candidate / "implementation.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("Candidate implementation missing or symlinked")
    banned = "harness.contracts.wp20_reference"
    for py_file in candidate.rglob("*.py"):
        if py_file.is_symlink():
            raise ValueError("Symlinked Python module in candidate")
        try:
            body = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise ValueError("Candidate contains invalid Python") from exc
        for node in ast.walk(body):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(banned):
                raise ValueError("Candidate imports official reference")
            if isinstance(node, ast.Import) and any(
                alias.name.startswith(banned) for alias in node.names
            ):
                raise ValueError("Candidate imports official reference")
    return source


def _matching_x18(observed, expected, git_evidence):
    """Validate semantic Git DAG instead of requiring identical commit timestamps."""
    if not isinstance(observed, dict) or not isinstance(git_evidence, dict):
        return False
    sha_keys = {"base_sha", "head_a_sha", "head_b_sha", "merge_sha", "merge_parents"}
    if {key: value for key, value in observed.items() if key not in sha_keys} != {
        key: value for key, value in expected.items() if key not in sha_keys
    }:
        return False
    for name in ("base_sha", "head_a_sha", "head_b_sha", "merge_sha"):
        sha = observed.get(name)
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            return False
        if sha != git_evidence.get(name):
            return False
    if len({observed[name] for name in ("base_sha", "head_a_sha", "head_b_sha", "merge_sha")}) != 4:
        return False
    parents = observed.get("merge_parents")
    return (
        isinstance(parents, list)
        and set(parents) == {observed["head_a_sha"], observed["head_b_sha"]}
        and parents == git_evidence.get("merge_parents")
        and observed["commits"] == git_evidence.get("commits")
        and git_evidence.get("clean") is True
    )


def verify_case(case, candidate):
    if case not in READY:
        raise ValueError("Unqualified case")
    candidate = Path(candidate).resolve(strict=True)
    if (
        candidate.name != case
        or not RUN_ID.fullmatch(candidate.parent.name)
        or not candidate.is_relative_to(RUNS.resolve())
    ):
        raise ValueError("Candidate outside approved case")
    source = _candidate_source(candidate)
    output = []
    for row in load_scenarios()["cases"][case]:
        expected = _reference(case, row["input"])
        with tempfile.TemporaryDirectory(prefix="wp20-candidate-") as wd:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": wd,
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
            }
            try:
                p = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(ROOT / "harness/contracts/wp20_worker.py"),
                        str(source),
                    ],
                    input=json.dumps(row["input"], ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    cwd=wd,
                    env=env,
                    timeout=8,
                    check=False,
                )
                received = json.loads(p.stdout)
                observed = received.get("output")
                passed = (
                    p.returncode == 0
                    and received.get("mutated") is False
                    and (
                        received.get("ok") is expected["ok"]
                        and received.get("kind") == expected["kind"]
                        and (
                            _matching_x18(
                                observed, expected["output"], received.get("git_evidence")
                            )
                            if case == "X18" and expected["ok"]
                            else observed == expected["output"]
                        )
                        and (
                            received.get("files") == expected["files"]
                            if expected["ok"]
                            else isinstance(received.get("files"), dict)
                            and all(name in expected["files"] for name in received["files"])
                        )
                    )
                )
            except (ValueError, subprocess.TimeoutExpired, json.JSONDecodeError):
                passed = False
            output.append({"scenario": row["id"], "passed": bool(passed)})
    return {
        "case": case,
        "checks": output,
        "passed": sum(x["passed"] for x in output),
        "failed": sum(not x["passed"] for x in output),
        "technical_only": True,
        "human_approved": False,
    }


def audit_scope(base_sha, head="HEAD"):
    if not re.fullmatch(r"[0-9a-f]{40}", base_sha) or (
        head != "HEAD" and not re.fullmatch(r"[0-9a-f]{40}", head)
    ):
        raise ValueError("Invalid commit SHA")
    diff = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", base_sha, head],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    bad = []
    for line in diff.stdout.splitlines():
        status, path = line.split("\t", 1)
        parts = Path(path).parts
        if not (
            status == "A"
            and len(parts) >= 7
            and parts[:4] == ("harness", "tests", "wp20", "runs")
            and RUN_ID.fullmatch(parts[4])
            and parts[5] in READY
        ):
            bad.append(path)
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-spec", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--head-sha")
    args = parser.parse_args()
    try:
        result = {"spec": check_spec()}
        if args.base:
            result["unauthorized_changes"] = audit_scope(args.base, args.head_sha or "HEAD")
        if args.case:
            if not args.run_dir:
                raise ValueError("Missing --run-dir")
            result["cases"] = [verify_case(args.case, args.run_dir)]
        elif args.scan:
            result["cases"] = []
            if RUNS.is_dir():
                for run in sorted(RUNS.iterdir()):
                    if not run.is_dir() or not RUN_ID.fullmatch(run.name):
                        continue
                    existing = {p.name for p in run.iterdir() if p.is_dir()}
                    for c in sorted(READY):
                        if c in existing:
                            result["cases"].append(verify_case(c, run / c))
                        else:
                            result["cases"].append({"case": c, "failed": 1, "missing": True})
        elif not args.check_spec and not args.base:
            raise ValueError("Select --check-spec, --case, --scan, or --base")
        if args.scan and not result.get("cases"):
            raise ValueError("No WP20 exam runs submitted; empty scan is not PASS")
        result["status"] = (
            "FAIL"
            if result.get("unauthorized_changes")
            or any(row["failed"] for row in result.get("cases", []))
            else "PASS"
        )
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
