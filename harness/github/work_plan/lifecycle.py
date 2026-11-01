"""Work Plan identity, Atlas links and completed-result contract.

Plan declarations reference existing authority; they grant no execution rights.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
BLOCK = re.compile(r"```workplan-json\s*\n(.*?)\n```", re.S)
HUMAN = re.compile(r"```from-human-json\s*\n(.*?)\n```", re.S)
STATUSES = frozenset(("PASS", "FAIL", "BLOCKED", "PARTIAL", "NOT_RUN", "WARNING", "HUMAN_DECISION"))


class PlanError(ValueError):
    """Unresolvable or inconsistent Work Plan."""


def source(root, name):
    """Reject filesystem traversal, external symlinks and nonexistent sources."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise PlanError("Unsafe source path")
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PlanError("Missing or escaping source " + name)
    return path


def block(text, pattern):
    parts = pattern.findall(text)
    if len(parts) != 1:
        raise PlanError("Exactly one typed declaration is required")
    try:
        obj = json.loads(parts[0])
    except json.JSONDecodeError as exc:
        raise PlanError("Malformed declaration JSON") from exc
    if not isinstance(obj, dict):
        raise PlanError("Expected JSON object")
    return obj


def ids(root, path, field):
    return {x["id"] for x in json.loads(source(root, path).read_text(encoding="utf-8"))[field]}


def check(plan_path, root=ROOT, check_contracts=True):
    """Validate Work Plan refs and DAG; this is never task authorization."""
    if (
        not plan_path.startswith("harness/github/work_plan/")
        or "/from_human/" in plan_path
        or not plan_path.endswith(".md")
    ):
        raise PlanError("Only a repository Work Plan can be dispatched")
    plan = block(source(root, plan_path).read_text(encoding="utf-8"), BLOCK)
    if plan.get("schema") != "saebase.workplan.v1" or not re.fullmatch(
        r"WP[0-9]+", str(plan.get("plan_id", ""))
    ):
        raise PlanError("Invalid Work Plan identity")
    if plan.get("status") not in ("draft", "reviewed", "approved"):
        raise PlanError("Missing review state")
    human = plan.get("from_human", "")
    if not isinstance(human, str) or not human.startswith("harness/github/work_plan/from_human/"):
        raise PlanError("Missing from_human reference")
    intent = block(source(root, human).read_text(encoding="utf-8"), HUMAN)
    if (
        intent.get("schema") != "saebase.from_human.v1"
        or intent.get("plan_id") != plan["plan_id"]
        or not intent.get("goal")
        or not intent.get("scientific_motivation")
    ):
        raise PlanError("Human intent is missing or mismatched")
    known = {
        "core_ids": ids(root, "harness/core/project_graph.json", "nodes"),
        "knowledge_ids": ids(root, "doc/knowledge/context_graph.json", "nodes"),
        "context_term_ids": (
            ids(root, "doc/knowledge/glossary.json", "terms")
            | ids(root, "doc/knowledge/atlas_terms.json", "terms")
        ),
    }
    atlas = plan.get("atlas") or {}
    for field, valid in known.items():
        entries = atlas.get(field)
        if (
            not isinstance(entries, list)
            or not entries
            or len(entries) != len(set(entries))
            or not set(entries) <= valid
        ):
            raise PlanError("Invalid Atlas " + field)
    paths = atlas.get("code_refs")
    if not isinstance(paths, list) or not paths:
        raise PlanError("Missing Code references")
    for path in paths:
        if not isinstance(path, str) or not path.endswith(".py"):
            raise PlanError("Code evidence must be Python source")
        source(root, path)
    study = plan.get("experiment") or {}
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(study.get("study_id", ""))):
        raise PlanError("Invalid study ID")
    config = study.get("config", "")
    if not isinstance(config, str) or not config.startswith("conf/experiment/"):
        raise PlanError("Research parameters must use conf/experiment/")
    source(root, config)
    seen = set()
    for step in plan.get("steps", []):
        if (
            not isinstance(step, dict)
            or step.get("id") in seen
            or not isinstance(step.get("id"), str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", step["id"]) is None
            or not isinstance(step.get("needs"), list)
            or any(not isinstance(dep, str) for dep in step["needs"])
            or not set(step["needs"]) <= seen
        ):
            raise PlanError("Invalid task DAG")
        allowed = {"local", "network", "gpu", "native", "publish"}
        permissions = step.get("permissions", [step.get("permission")])
        if (
            not isinstance(permissions, list)
            or not permissions
            or any(type(p) is not str or p not in allowed for p in permissions)
            or len(permissions) != len(set(permissions))
            or not step.get("goal")
            or not step.get("evidence")
        ):
            raise PlanError("Missing purpose, evidence or valid permissions")
        if step.get("on_failure", "global_block") not in ("local_block", "global_block"):
            raise PlanError("Unsupported failure policy")
        if plan["status"] == "approved" and not step.get("command"):
            raise PlanError("Approved executable step requires a concrete command")
        seen.add(step["id"])
    if not seen or not plan.get("deliverables"):
        raise PlanError("Steps and deliverables are mandatory")
    for row in plan["deliverables"]:
        if (
            not isinstance(row, dict)
            or not row.get("experiment")
            or not row.get("metric")
            or not row.get("expected")
        ):
            raise PlanError("Incomplete result template")
    if check_contracts:
        from harness.contracts.registry import validate_plan

        validate_plan(plan_path)
    return plan


def completed(plan_path, receipt, root=ROOT):
    """Fill a new copy of the original plan from observed results."""
    plan = check(plan_path, root=root, check_contracts=(root.resolve() == ROOT.resolve()))
    if (
        receipt.get("schema") != "saebase.workplan.run.v1"
        or receipt.get("plan_id") != plan["plan_id"]
        or not receipt.get("run_id")
    ):
        raise PlanError("Mismatched run receipt")
    observations = receipt.get("results", {})
    if not isinstance(observations, dict) or set(observations) - {
        x["experiment"] for x in plan["deliverables"]
    }:
        raise PlanError("Unexpected result")
    rows = [
        "| Experiment | Metric | Expected | Actual | Status | Notes | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in plan["deliverables"]:
        result = observations.get(entry["experiment"], {"status": "NOT_RUN"})
        status = result.get("status")
        if status not in STATUSES or (
            status == "PASS" and (result.get("value") is None or not result.get("evidence"))
        ):
            raise PlanError("Results must have truthful status and evidence")
        values = (
            entry["experiment"],
            entry["metric"],
            entry["expected"],
            result.get("value", ""),
            status,
            result.get("notes", ""),
            result.get("evidence", ""),
        )
        rows.append(
            "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |"
        )
    markdown = source(root, plan_path).read_text(encoding="utf-8").rstrip()
    return (
        markdown
        + "\n\n## Filled execution results\n\n"
        + "\n".join(rows)
        + "\n\nRun ID: "
        + str(receipt["run_id"])
        + "\nSource commit: "
        + str(receipt.get("source_commit", "UNVERIFIED"))
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("validate", "inspect", "complete"))
    p.add_argument("--plan", required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        plan = check(args.plan)
        if args.action != "complete":
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            if args.receipt is None or args.output is None:
                raise PlanError("Observed receipt and output path required")
            out = args.output.resolve()
            if not out.is_relative_to((ROOT / "experiment").resolve()):
                raise PlanError("Completed plans belong in experiment/ artifacts")
            content = completed(args.plan, json.loads(args.receipt.read_text(encoding="utf-8")))
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x", encoding="utf-8") as f:
                f.write(content)
            print(json.dumps({"status": "filled", "path": str(out)}))
        return 0
    except (PlanError, ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(2, "WORK PLAN BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
