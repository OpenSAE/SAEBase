"""Load predeclared Harness contracts and validate a Work Plan's selection.

Work plans only reference stable contract IDs; they must not redefine or
silently waive global invariants. This module is read-only and stdlib-only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "harness/contracts/catalog.json"
PLAN_ROOT = ROOT / "harness/github_work_plan"
BLOCK = re.compile(r"```harness-contracts-json\s*\n(.*?)\n```", re.DOTALL)
CASE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,63}\Z")


class ContractError(ValueError):
    """A contract selection or catalog is invalid and must not be activated."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def catalog() -> dict:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    _need(data.get("schema") == "saebase.harness.contract_registry.v1", "Unknown catalog schema")
    rows = data.get("contracts")
    _need(isinstance(rows, list) and rows, "No contracts in catalog")
    ids = [row["id"] for row in rows]
    _need(len(set(ids)) == len(ids), "Duplicate contract ID")
    for row in rows:
        _need(re.fullmatch(r"CT-[A-Z][A-Z0-9-]*", row["id"]) is not None,
              "Invalid contract ID")
        _need(row["enforcement"] in {"always", "conditional"}, "Invalid enforcement")
        _need(isinstance(row["requirements"], list) and bool(row["requirements"]),
              "Contract without executable obligations")
    always = data["required_always"]
    _need(len(set(always)) == len(always) and set(always) ==
          {row["id"] for row in rows if row["enforcement"] == "always"},
          "Always-on contracts are inconsistent")
    _need(len(set(data["task_tags"])) == len(data["task_tags"]),
          "Duplicate category tag")
    return data


def plan_path(relative: str) -> Path:
    _need(isinstance(relative, str) and relative.endswith(".md") and "\\" not in relative,
          "Expected a repository-relative Markdown Work Plan")
    _need(not relative.startswith("/") and ".." not in PurePosixPath(relative).parts,
          "Unsafe Work Plan path")
    path = (ROOT / relative).resolve()
    _need(path.is_relative_to(PLAN_ROOT.resolve()) and path.is_file(),
          "Work Plan is missing or outside the approved plan directory")
    return path


def declaration(relative: str) -> dict:
    text = plan_path(relative).read_text(encoding="utf-8")
    matches = BLOCK.findall(text)
    _need(len(matches) == 1, "Exactly one harness-contracts-json declaration required")
    try:
        obj = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ContractError("Invalid contract declaration JSON") from exc
    _need(isinstance(obj, dict) and set(obj) == {"registry", "plan", "cases"},
          "Invalid declaration fields")
    _need(obj["registry"] == catalog()["schema"], "Wrong contract registry")
    _need(isinstance(obj["cases"], dict) and bool(obj["cases"]),
          "Work Plan must declare each case")
    return obj


def _ids(value: object, known: set[str], label: str) -> list[str]:
    _need(isinstance(value, list) and bool(value) and
          all(isinstance(v, str) for v in value), label + " must list contract IDs")
    _need(len(set(value)) == len(value), "Duplicate contract: " + label)
    _need(set(value) <= known, "Unknown contract ID in " + label + ": " +
          repr(sorted(set(value) - known)))
    return list(value)


def resolve(relative: str, case: str) -> dict:
    """Resolve immutable universal obligations + declared task-specific contracts."""
    data = catalog()
    selection = declaration(relative)
    known = {x["id"]: x for x in data["contracts"]}
    _need(CASE.fullmatch(case) is not None, "Invalid case identifier")
    _need(case in selection["cases"], "Case was not declared in the Work Plan")
    plan_ids = _ids(selection["plan"], set(known), "plan")
    row = selection["cases"][case]
    _need(isinstance(row, dict) and set(row) == {"tags", "contracts"},
          "A case must explicitly declare tags and contracts")
    tags = row["tags"]
    _need(isinstance(tags, list) and bool(tags) and
          all(isinstance(x, str) for x in tags) and len(set(tags)) == len(tags) and
          set(tags) <= set(data["task_tags"]), "Invalid or undeclared task tags")
    case_ids = _ids(row["contracts"], set(known), case)
    selected = list(dict.fromkeys(data["required_always"] + plan_ids + case_ids))
    for tag in tags:
        # Each declared applicable tag needs an explicitly selected
        # conditional contract with this tag, not only a global generic rule.
        _need(any(known[ident]["enforcement"] == "conditional" and
                  tag in known[ident]["tags"] for ident in selected),
              "No specific contract covers tag: " + tag)
    return {
        "schema": "saebase.harness.selected_contracts.v1",
        "registry_version": data["version"],
        "plan": relative,
        "case": case,
        "tags": tags,
        "contract_ids": selected,
        "requirements": [
            {"id": ident, "title": known[ident]["title"],
             "requirements": known[ident]["requirements"],
             "negative_examples": known[ident]["negative_examples"]}
            for ident in selected
        ],
    }


def validate_plan(relative: str) -> dict:
    obj = declaration(relative)
    resolved = {case: resolve(relative, case) for case in obj["cases"]}
    return {"plan": relative, "cases": sorted(resolved),
            "case_contracts": {k: v["contract_ids"] for k, v in resolved.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--plan", required=True)
    show = commands.add_parser("resolve")
    show.add_argument("--plan", required=True)
    show.add_argument("--case", required=True)
    commands.add_parser("list")
    args = parser.parse_args()
    try:
        if args.command == "list":
            result = {"version": catalog()["version"],
                      "contract_ids": [r["id"] for r in catalog()["contracts"]]}
        elif args.command == "check":
            result = validate_plan(args.plan)
        else:
            result = resolve(args.plan, args.case)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ContractError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, "CONTRACT INVALID: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
