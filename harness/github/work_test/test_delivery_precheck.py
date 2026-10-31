"""Delivery routing regression; no network or mutating Git commands."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/workplan/delivery_precheck.py"
spec = importlib.util.spec_from_file_location("delivery_precheck", SCRIPT)
assert spec and spec.loader
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


def test_profiles_are_scope_based() -> None:
    assert delivery.choose_profile(["README.md"], "auto") == "documentation"
    assert delivery.choose_profile(["tasks/benchmark/new.py"], "auto") == "code"
    assert delivery.choose_profile(["scripts/experiments/qa500/run.py"], "auto") == "experiment"
    assert delivery.choose_profile(["harness/contracts/catalog.json"], "auto") == "governance"
    assert delivery.choose_profile([], "auto") == "unknown"


def test_path_validation_blocks_traversal() -> None:
    for bad in ("../outside", "/absolute", "a//b", "a/./b", "a\\b", "a/*.py"):
        try:
            delivery.validate_path(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe path passed: " + bad)
    assert delivery.validate_path("scripts/workplan/example.py") == (
        "scripts/workplan/example.py"
    )


def test_no_unapproved_execution_in_plan_mode() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "plan", "--changed", "README.md"],
        cwd=delivery.ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    import json
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["profile"] == "documentation"
    assert all(check["status"] == "NOT_RUN" for check in report["checks"])
    assert report["limits"].startswith("Read-only")


def test_local_execution_requires_order() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "local",
         "--changed", "tasks/benchmark/demo.py"],
        cwd=delivery.ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert '"status": "BLOCKED"' in result.stdout


def test_safe_command_allowlist() -> None:
    assert all(cmd[0] == sys.executable for cmd in delivery.SAFE.values())
    assert not any("install" in cmd or "fetch" in cmd for cmd in delivery.SAFE.values())
    assert "authorized-real-pilot" not in delivery.SAFE
