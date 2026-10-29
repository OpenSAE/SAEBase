"""Independent WP18 acceptance gate; never grants human approval or production authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.contracts import invariants, registry, trace
from harness.contracts.selection import audit_selection

ROOT = registry.ROOT
ACCEPTANCE = ROOT / "harness/tests/wp18/acceptance"


def review(*, plan: str, case: str, receipt_path: Path, evaluator_path: Path) -> dict:
    """Evaluator-controlled records are kept in the protected acceptance subtree."""
    selected = registry.resolve(plan, case)
    evaluator_path = evaluator_path.resolve(strict=True)
    if not evaluator_path.is_relative_to(ACCEPTANCE.resolve()):
        raise registry.ContractError("Independent evaluator evidence must be under acceptance/")
    record = json.loads(evaluator_path.read_text(encoding="utf-8"))
    receipt_path = receipt_path.resolve(strict=True)
    if not receipt_path.is_relative_to((ROOT / "harness/tests/wp18/runs").resolve()):
        raise registry.ContractError("Candidate receipt must be in a WP18 run")
    receipt = trace.check_receipt_file(receipt_path, plan=plan, case=case)
    issues = []
    if (
        record.get("case_id") != case
        or record.get("schema") != "saebase.harness.independent_evaluation.v1"
    ):
        issues.append("Invalid independent evaluation identity")
    signals = record.get("task_signals")
    if not isinstance(signals, list):
        issues.append("Missing independently reviewed task IO signals")
    else:
        try:
            missing = audit_selection(selected["contract_ids"], signals)
            if missing:
                issues.append("SELECTION_GAP: " + ",".join(missing))
        except registry.ContractError as exc:
            issues.append(str(exc))
    tests = record.get("tests")
    if (
        not isinstance(tests, list)
        or not tests
        or not all(
            isinstance(t, dict) and isinstance(t.get("name"), str) and type(t.get("passed")) is bool
            for t in tests
        )
    ):
        issues.append("Missing independent positive/negative test results")
    elif any(not t["passed"] for t in tests):
        issues.append("Independent test failure")
    if record.get("source_verified") is not True:
        issues.append("Source identity/AST not independently verified")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        issues.append("Missing independently inspected artifacts")
    else:
        for i, item in enumerate(artifacts):
            try:
                if not isinstance(item, dict):
                    raise invariants.InvariantError("Invalid artifact")
                kind, payload = item.get("kind"), item.get("payload")
                if kind == "source_reference":
                    invariants.reference_location(payload, ROOT)
                elif kind == "structured_output":
                    cleaned, redactions = invariants.redact_structure(payload)
                    if redactions or cleaned != payload:
                        issues.append(f"Artifact {i} contains unredacted sensitive information")
                elif kind == "tree":
                    invariants.validated_dfs(payload)
                elif kind == "bilingual_record":
                    defects = invariants.localization_defects(payload)
                    issues.extend(f"Artifact {i}: {x}" for x in defects)
                else:
                    issues.append(f"Unsupported artifact kind at {i}")
            except (ValueError, OSError, TypeError) as exc:
                issues.append(f"Artifact {i}: {exc}")
    return {
        "case": case,
        "status": "FAIL" if issues else "TECHNICAL_CHECKS_PASSED_NEEDS_HUMAN_REVIEW",
        "issues": issues,
        "contract_count": receipt["contract_count"],
        "human_approval": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()
    try:
        verdict = review(
            plan=args.plan, case=args.case, receipt_path=args.receipt, evaluator_path=args.evaluator
        )
    except (registry.ContractError, ValueError, OSError, TypeError) as exc:
        verdict = {
            "case": args.case,
            "status": "BLOCKED",
            "issues": [str(exc)],
            "human_approval": False,
        }
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["status"] == "TECHNICAL_CHECKS_PASSED_NEEDS_HUMAN_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
