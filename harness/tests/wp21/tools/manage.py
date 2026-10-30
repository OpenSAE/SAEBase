"""Trusted WP21 dispatcher and read-only scope audit. Standard library only.

Run this copy from the immutable package, not from candidate-edited code.
Preparation never runs a candidate, downloads a model, or changes GitHub.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALWAYS, BASE_COMMIT, IGNORED_DIRS, ROOT, SUITE, allowed_source,
                    canonical_hash, contained, get_case, load_spec, read_json,
                    relative_name, sha256, snapshot, source_changes, write_json, validate_suite_lock)


def check_spec(root: Path = ROOT) -> dict:
    integrity = validate_suite_lock(root)
    spec = load_spec(root)
    if spec["schema"] != "saebase.wp21.cases.v1" or spec["base_commit"] != BASE_COMMIT:
        raise ValueError("wrong suite schema/base revision")
    cases = spec["cases"]
    if [c["id"] for c in cases] != [f"R{i:02d}" for i in range(1, 21)]:
        raise ValueError("exactly R01..R20, ordered and unique, required")
    catalog = read_json(root / "harness/contracts/catalog.json")
    ids = {c["id"] for c in catalog["contracts"]}
    tag_set = set(catalog["task_tags"])
    seed_count = 0
    for case in cases:
        cid = case["id"]
        contracts = case["contracts"]
        if len(contracts) != len(set(contracts)) or not ALWAYS <= set(contracts) or not set(contracts) <= ids:
            raise ValueError(cid + ": invalid contract binding")
        if not set(case["tags"]) <= tag_set:
            raise ValueError(cid + ": unknown tag")
        for key in ["goal", "steps", "positive_checks", "negative_checks", "required_artifacts", "entrypoints"]:
            if not case.get(key):
                raise ValueError(cid + ": missing " + key)
        if case.get("profile") not in {"gemma_inference", "pythia14m_inference", "gemma_training"}:
            raise ValueError(cid + ": unknown environment profile")
        for key in ["max_prompts", "max_tokens_per_prompt", "max_trials", "max_seconds", "max_new_artifact_mb"]:
            if type(case.get("budget", {}).get(key)) is not int or case["budget"][key] < 1:
                raise ValueError(cid + ": positive integer budget required: " + key)
        for name in case["allowed_paths"]:
            relative_name(name, pattern=True)
            if name.startswith(("harness/", ".cursor/", ".github/", "saebase/runtime/", "doc/")) or "/" not in name:
                raise ValueError(cid + ": protected or root-level product edit")
        for name in case["context_paths"]:
            if not contained(root, name, exists=True).is_file():
                raise ValueError(cid + ": missing source context " + name)
        if not (root / f"harness/tests/wp21/cases/{cid}.md").is_file():
            raise ValueError(cid + ": missing task card")
        seed = case.get("seed_fixture")
        if seed:
            relative_name(seed)
            manifest_path = root / "harness/tests/wp21/fixtures" / seed / "manifest.json"
            manifest = read_json(manifest_path)
            if manifest.get("case_id") != cid or manifest.get("base_commit") != BASE_COMMIT:
                raise ValueError(cid + ": seed identity/base mismatch")
            replaced = {x["path"]: x["base_sha256"] for x in manifest.get("replaced_base_paths", [])}
            destinations = set()
            for file in manifest["files"]:
                src = contained(manifest_path.parent, file["source"], exists=True)
                dest = file["destination"]
                if dest in destinations:
                    raise ValueError(cid + ": duplicate seed destination")
                destinations.add(dest)
                if not allowed_source(dest, case):
                    raise ValueError(cid + ": seed outside case scope " + dest)
                if file.get("sha256") != sha256(src):
                    raise ValueError(cid + ": seed checksum mismatch " + dest)
                baseline = contained(root, dest)
                if baseline.is_file() and replaced.get(dest) != sha256(baseline):
                    raise ValueError(cid + ": seed replacement not bound to exact baseline file " + dest)
                if not baseline.exists() and dest in replaced:
                    raise ValueError(cid + ": declared seed replacement target missing")
            if not set(replaced) <= destinations:
                raise ValueError(cid + ": replacement declaration outside seed files")
            seed_count += 1
    return {"status": "SPEC_VALID", "cases": len(cases), "seeded_cases": seed_count,
            "base_commit": BASE_COMMIT, "runtime_results": "NOT_RUN", "suite_integrity": integrity,
            "meaning": "Specification integrity only. Independent real-model acceptance is separate."}


def prepare(destination: Path, case_ids: list[str], run_id: str, assignment_ref: str | None) -> dict:
    check_spec()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,47}", run_id) is None:
        raise ValueError("run_id must have 3..48 safe characters")
    destination = destination.absolute()
    if destination.exists() or destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("choose a NEW directory outside the package source")
    destination.mkdir(parents=True)
    created = []
    for cid in case_ids:
        case = get_case(cid)
        candidate = destination / "candidates" / cid
        def ignore(folder, names):
            skipped = {name for name in names if name in IGNORED_DIRS or name.endswith((".pyc", ".pyo"))}
            if Path(folder) == ROOT:
                skipped.update({"experiment", "experiments"} & set(names))
            if Path(folder) == SUITE:
                skipped.update({"runs", "acceptance"} & set(names))
            return skipped
        shutil.copytree(ROOT, candidate, ignore=ignore)
        seed = case.get("seed_fixture")
        if seed:
            seedroot = SUITE / "fixtures" / seed
            for file in read_json(seedroot / "manifest.json")["files"]:
                source = contained(seedroot, file["source"], exists=True)
                target = contained(candidate, file["destination"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        # Preserve current runtime's singular directory. No code/layout migration.
        (candidate / "experiment").mkdir()
        evidence_dir = candidate / "harness/tests/wp21/runs" / run_id / cid
        evidence_dir.mkdir(parents=True)
        (candidate / "tests/wp21" / cid).mkdir(parents=True, exist_ok=True)
        before = snapshot(candidate)
        control = destination / "control" / cid
        write_json(control / "source_before.json", before)
        assignment = {
            "schema": "saebase.wp21.assignment.v1", "case_id": cid, "run_id": run_id,
            "base_commit": BASE_COMMIT, "case_sha256": canonical_hash(case),
            "spec_sha256": sha256(candidate / "harness/tests/wp21/cases.json"),
            "suite_lock_sha256": sha256(candidate / "harness/tests/wp21/suite_lock.json"),
            "catalog_sha256": sha256(candidate / "harness/contracts/catalog.json"),
            "protocol_sha256": sha256(candidate / "harness/contracts/REAL_WORKFLOW_PROTOCOL.md"),
            "source_before_sha256": canonical_hash(before),
            "state": "assigned" if assignment_ref else "prepared_not_assigned",
            "assignment_ref": assignment_ref,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "allowed_paths": case["allowed_paths"], "budget": case["budget"],
            "evidence_root": evidence_dir.relative_to(candidate).as_posix(),
            "runtime_output_root": "experiment",
            "note": "Dispatcher record of an existing user assignment; not a fabricated human approval, grading result, or production/release permission.",
        }
        write_json(control / "assignment.json", assignment)
        write_json(candidate / ".cursor/wp21_assignment.json", assignment)
        # These are templates. They cannot pass an evidence audit unchanged.
        write_json(evidence_dir / "evidence.json", {
            "schema": "saebase.wp21.evidence.v1", "case_id": cid, "run_id": run_id,
            "status": "NOT_RUN", "evidence_kind": "unexecuted", "base_commit": BASE_COMMIT,
            "source_snapshot_sha256": None, "input_identity": None, "model_identity": None,
            "sae_identity": None, "environment": None, "runtime_config_sha256": None,
            "canonical_config_sha256": None, "executed_result_paths": [], "command_refs": [],
            "unresolved": ["Real checkpoint execution has not started."],
        })
        write_json(evidence_dir / "contract_receipt.json", {
            "schema": "saebase.wp21.contract_receipt.v1", "case_id": cid,
            "registry_version": "1.1.0", "contract_ids": case["contracts"],
            "context_paths": [], "edited_paths": [], "commands": [], "unresolved": ["NOT_RUN"],
        })
        write_json(evidence_dir / "selfcheck.json", {"status": "NOT_RUN", "checks": []})
        write_json(evidence_dir / "expected_code_diff.json", {"status": "NOT_WRITTEN", "changes": []})
        write_json(evidence_dir / "review_candidate.json", {"status": "unchecked", "nodes": [], "context_terms": []})
        (evidence_dir / "report.md").write_text("# " + cid + " 执行报告\n\n当前状态 NOT_RUN。填写实际修改、真实运行、失败、证据和未完成项。\n", encoding="utf-8")
        created.append({"case_id": cid, "workspace": str(candidate), "evidence": str(evidence_dir), "state": assignment["state"]})
    write_json(destination / "dispatch.json", {"schema": "saebase.wp21.dispatch.v1", "run_id": run_id, "cases": created})
    return {"status": "PREPARED", "cases": created, "models_executed": 0}


def scope_audit(workspace: Path, control: Path) -> dict:
    assignment = read_json(control / "assignment.json")
    cid = assignment["case_id"]
    case = get_case(cid)  # trusted package copy, never candidate catalog
    before = read_json(control / "source_before.json")
    if canonical_hash(before) != assignment["source_before_sha256"]:
        raise ValueError("trusted before-source snapshot is corrupt")
    # Recheck trusted assignment bindings from the immutable teacher source.
    if assignment.get("case_sha256") != canonical_hash(case):
        raise ValueError("assignment case identity no longer matches trusted exam")
    if assignment.get("spec_sha256") != sha256(ROOT / "harness/tests/wp21/cases.json"):
        raise ValueError("assignment references another frozen exam")
    if assignment.get("suite_lock_sha256") is not None and assignment["suite_lock_sha256"] != sha256(ROOT / "harness/tests/wp21/suite_lock.json"):
        raise ValueError("assignment references another suite lock")
    after = snapshot(workspace)
    changes = source_changes(before, after)
    violations = [x["path"] for x in changes if not allowed_source(x["path"], case)]
    copied_assignment = read_json(workspace / ".cursor/wp21_assignment.json")
    if copied_assignment != assignment:
        violations.append(".cursor/wp21_assignment.json")
    if case != get_case(cid, workspace):
        violations.append("harness/tests/wp21/cases.json")
    return {"status": "SCOPE_VALID" if not violations else "SCOPE_VIOLATION",
            "case_id": cid, "changes": changes, "violations": violations,
            "source_snapshot_sha256": canonical_hash(after), "source_files": after,
            "note": "Scope correctness does not establish functional correctness or real execution."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-spec")
    prep = sub.add_parser("prepare")
    prep.add_argument("--destination", required=True, type=Path)
    prep.add_argument("--case", default="all")
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--assignment-ref", help="Trusted dispatcher only: cite the ACTUAL existing user assignment. Omit for preparation only.")
    audit = sub.add_parser("scope")
    audit.add_argument("--workspace", required=True, type=Path)
    audit.add_argument("--control", required=True, type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "check-spec":
            result = check_spec()
        elif args.command == "prepare":
            cases = [f"R{i:02d}" for i in range(1, 21)] if args.case == "all" else [args.case]
            result = prepare(args.destination, cases, args.run_id, args.assignment_ref)
        else:
            result = scope_audit(args.workspace.resolve(), args.control.resolve())
            if args.output:
                if args.output.resolve().is_relative_to(args.workspace.resolve()):
                    raise ValueError("reviewer output must be outside candidate workspace")
                write_json(args.output, result, exclusive=True)
        printable = {k: v for k, v in result.items() if k != "source_files"}
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return int(result.get("status") == "SCOPE_VIOLATION")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps({"status": "INVALID", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
