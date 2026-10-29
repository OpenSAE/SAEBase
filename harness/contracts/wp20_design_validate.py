"""Check WP20 task design and predeclared contract binding, not candidate correctness.

This intentionally cannot return an exam PASS. Reviewer-owned executable
oracles must be implemented and qualified before any case becomes READY.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from harness.contracts import registry

ROOT = registry.ROOT
PLAN = "harness/github_work_plan/20_integrated_multistage_harness_exams.md"
SPEC = ROOT / "harness/contracts/wp20_cases.json"
CASE_IDS = [f"X{i:02d}" for i in range(1, 21)]
CASE_GROUPS = {
    "Harness and Atlas",
    "Scientific data and SAE",
    "Evaluation and reliability",
    "Release and multi-agent",
}


def load_spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def validate(data: dict | None = None) -> dict:
    spec = load_spec() if data is None else data
    if not isinstance(spec, dict) or spec.get("schema") != "saebase.wp20.integrated_exam_design.v1":
        raise ValueError("Invalid WP20 design schema")
    cases = spec.get("cases")
    if not isinstance(cases, list) or [c.get("id") for c in cases] != CASE_IDS:
        raise ValueError("WP20 must declare exactly X01-X20 in order")
    if spec.get("status") != "SPEC_REVIEWED" or spec.get("execution_contract", {}).get("formal_approval") != "reviewer only":
        raise ValueError("WP20 is a design only, not independent acceptance")
    groups = {c["group"] for c in cases}
    if groups != CASE_GROUPS or any(sum(c["group"] == g for c in cases) != 5 for g in CASE_GROUPS):
        raise ValueError("Expected four distinct groups of five exams")
    if len({c["title"] for c in cases}) != len(cases):
        raise ValueError("Duplicate exam title")
    if len({c["goal"] for c in cases}) != len(cases):
        raise ValueError("Duplicate exam objective")
    checked_sources = 0
    for case in cases:
        if case.get("status") != "SPEC_REVIEWED" or case.get("oracle_status") != "NOT_IMPLEMENTED":
            raise ValueError(f"{case['id']} incorrectly claims an executable verifier")
        expected = registry.resolve(PLAN, case["id"])
        if expected["tags"] != case["tags"] or set(expected["contract_ids"]) != set(
            registry.catalog()["required_always"] + ["CT-ERROR"] + case["contracts"]
        ):
            raise ValueError(f"{case['id']} Contract binding mismatch")
        if not isinstance(case.get("stages"), list) or len(case["stages"]) < 4:
            raise ValueError(f"{case['id']} must require real staged execution")
        if [s["number"] for s in case["stages"]] != list(range(1, len(case["stages"]) + 1)):
            raise ValueError(f"{case['id']} invalid execution sequence")
        for key in ("positive_cases", "negative_cases"):
            if not isinstance(case.get(key), list) or len(case[key]) < 2:
                raise ValueError(f"{case['id']} needs two distinct {key}")
            if len(set(case[key])) != len(case[key]):
                raise ValueError(f"{case['id']} duplicate {key}")
        for key in ("goal", "metamorphic", "oracle_plan"):
            if not isinstance(case.get(key), str) or len(case[key].strip()) < 25:
                raise ValueError(f"{case['id']} missing behavioral {key}")
        refs = case.get("source_refs")
        if not isinstance(refs, list) or len(refs) < 2 or len(refs) != len(set(refs)):
            raise ValueError(f"{case['id']} needs multiple real code references")
        for ref in refs:
            if not isinstance(ref, str) or not ref.startswith(("harness/", "saebase/", ".github/", ".cursor/")):
                raise ValueError("Untrusted source reference")
            if ref.startswith("/") or ".." in PurePosixPath(ref).parts or "\\" in ref:
                raise ValueError("Noncanonical source reference")
            target = (ROOT / ref).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.exists():
                raise ValueError("WP20 source ref missing: " + ref)
            checked_sources += 1
        outputs = case.get("required_outputs")
        if not isinstance(outputs, list) or len(outputs) < 4 or len(outputs) != len(set(outputs)):
            raise ValueError(f"{case['id']} missing concrete outputs")
        budget = case.get("resource_budget", {})
        if (
            budget.get("device") != "CPU"
            or budget.get("network") is not False
            or budget.get("gpu") is not False
            or type(budget.get("wall_seconds")) is not int
            or budget["wall_seconds"] > 120
            or budget["wall_seconds"] <= 0
            or type(budget.get("memory_mb")) is not int
            or budget["memory_mb"] > 1024
        ):
            raise ValueError(f"{case['id']} resource budget violates contract")
    return {
        "schema": spec["schema"],
        "cases": len(cases),
        "groups": len(groups),
        "source_references_checked": checked_sources,
        "stage_count": sum(len(c["stages"]) for c in cases),
        "normal_scenarios": sum(len(c["positive_cases"]) for c in cases),
        "negative_scenarios": sum(len(c["negative_cases"]) for c in cases),
        "status": "SPEC_REVIEWED_ONLY_NOT_READY",
    }


def main() -> int:
    result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
