"""Read-only precheck routing for a SAEBase Work Plan and local-agent handoff.

No installer, fetch, checkout, mutation, download, model execution, or publish.
This program reports what needs verification; it does not grant authorization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("auto", "documentation", "code", "experiment", "governance", "release")


def git(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.returncode, result.stdout.strip()


def validate_path(path: str) -> str:
    if (
        not path or path.startswith("/") or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
        or any(c in path for c in "*?[]")
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError("Invalid relative source path: " + path)
    return path


def choose_profile(paths: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    if not paths:
        return "unknown"
    if any(p.startswith((".github/workflows/", ".cursor/", "harness/core/",
                         "harness/contracts/")) for p in paths):
        return "governance"
    if any(p.startswith(("scripts/experiments/", "conf/experiment/", "experiment/"))
           for p in paths):
        return "experiment"
    if any(p.startswith(("datasets/", "models/", "tasks/", "saebase/runtime/"))
           for p in paths):
        return "code"
    if any(p.startswith(("harness/visualization/", "atlas/")) for p in paths):
        return "code"
    if all(p.endswith((".md", ".rst", ".txt")) for p in paths):
        return "documentation"
    if any(p in ("pyproject.toml", "MANIFEST.in", "setup.py") for p in paths):
        return "release"
    return "code"


def recommended(profile: str) -> list[str]:
    checks = {
        "documentation": ["link-check", "scope-review"],
        "code": ["syntax-and-architecture", "dependency-check", "targeted-tests"],
        "experiment": ["syntax-and-architecture", "dependency-check",
                       "config-and-asset-identity", "bounded-cpu-pilot",
                       "authorized-real-pilot"],
        "governance": ["architecture-and-contracts", "independent-maintainer-review",
                       "targeted-regression"],
        "release": ["package-and-provenance", "exact-main-sha-ci",
                    "independent-release-approval"],
        "unknown": ["identify-scope-and-owner"],
    }
    return checks[profile]


# Only these allowlisted commands may be run by --run-safe. No shell.
SAFE = {
    "link-check": [sys.executable, "saebase/tools/check_documentation.py", "--check"],
    "syntax-and-architecture": [
        sys.executable, ".cursor/hooks/validate_repo.py", "--full"
    ],
    "dependency-check": [sys.executable, "-m", "pip", "check"],
    "architecture-and-contracts": [
        sys.executable, ".cursor/hooks/validate_repo.py", "--full"
    ],
}
SAFE_TIMEOUT = 120


def run_safe(checks: list[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for check in checks:
        if check not in SAFE:
            output.append({"check": check, "status": "NOT_RUN",
                           "reason": "Requires targeted scope, environment, or approval"})
            continue
        command = SAFE[check]
        try:
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                check=False, timeout=SAFE_TIMEOUT
            )
            output.append({
                "check": check, "status": "PASS" if completed.returncode == 0 else "FAIL",
                "command": command, "exit_code": completed.returncode,
                "output_tail": (completed.stdout + completed.stderr)[-2500:],
            })
        except (OSError, subprocess.TimeoutExpired) as exc:
            output.append({"check": check, "status": "BLOCKED",
                           "reason": type(exc).__name__})
    return output


def evaluate(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    problems: list[str] = []
    code, branch = git("branch", "--show-current")
    if code != 0:
        problems.append("Git repository identity unavailable")
    _, head = git("rev-parse", "HEAD")
    if args.expected_branch and branch != args.expected_branch:
        problems.append("Branch mismatch: expected " + args.expected_branch)
    if args.expected_sha and head != args.expected_sha:
        problems.append("HEAD mismatch: expected " + args.expected_sha)
    paths = sorted(set(validate_path(path) for path in args.changed))
    profile = choose_profile(paths, args.profile)
    checks = recommended(profile)
    order_status = "NOT_RUN"
    if args.work_order:
        validator = ROOT / ".cursor/skills/workflow/run-work-order/scripts/validate_work_order.py"
        try:
            result = subprocess.run(
                [sys.executable, str(validator), args.work_order],
                cwd=ROOT, capture_output=True, text=True, timeout=SAFE_TIMEOUT,
                check=False,
            )
            order_status = "PASS" if result.returncode == 0 else "FAIL"
            if result.returncode:
                problems.append("Work Order validation failed: " +
                                (result.stderr or result.stdout)[-600:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            order_status = "BLOCKED"
            problems.append("Work Order validator unavailable: " + type(exc).__name__)
    elif args.mode == "local":
        problems.append("Local production execution needs an authorized Work Order")
        order_status = "BLOCKED"
    measurements = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    }
    outcome = "BLOCKED" if problems else "PASS"
    results = run_safe(checks) if args.run_safe and not problems else [
        {"check": check, "status": "NOT_RUN",
         "reason": "Recommended only; no execution requested"} for check in checks
    ]
    if any(r["status"] == "FAIL" for r in results):
        outcome = "FAIL"
    elif any(r["status"] == "BLOCKED" for r in results):
        outcome = "BLOCKED"
    return ({
        "schema": "saebase.delivery.precheck.v1",
        "mode": args.mode, "status": outcome,
        "source": {"branch": branch, "head_sha": head, "expected_sha": args.expected_sha},
        "profile": profile, "changed_paths": paths,
        "recommended_checks": checks, "checks": results,
        "work_order_validation": order_status, "measurements": measurements,
        "blockers": problems,
        "limits": "Read-only route and safe checks only; no authorization or scientific acceptance",
    }, 0 if outcome == "PASS" else 2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("plan", "local"), default="plan")
    p.add_argument("--profile", choices=PROFILES, default="auto")
    p.add_argument("--changed", action="append", default=[], metavar="RELATIVE_PATH")
    p.add_argument("--expected-branch", default=None)
    p.add_argument("--expected-sha", default=None)
    p.add_argument("--work-order", default=None)
    p.add_argument("--run-safe", action="store_true")
    args = p.parse_args()
    try:
        report, code = evaluate(args)
    except (ValueError, OSError) as exc:
        print(json.dumps({"schema": "saebase.delivery.precheck.v1",
                          "status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
