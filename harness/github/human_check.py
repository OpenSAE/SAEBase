"""Pre-PR Human Check: Atlas-backed, independent, proposal-only decisions.

A proposal and its report live under ignored experiment/ until authorized.
No decision here changes an Atlas authority graph or scientific result.
This application-level gate supplements, never replaces, GitHub permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from harness.core import agent_flow as atlas

ROOT = atlas.ROOT
WORKSPACE = ROOT / "experiment" / "_human_check"
AUTHORITIES = (
    "harness/core/project_graph.json",
    "doc/knowledge/context_graph.json",
    "doc/code/code_tree.json",
    "doc/knowledge/glossary.json",
    "doc/knowledge/atlas_terms.json",
    "harness/environment/example.json",
)
SCHEMA = "saebase.human_check.v1"
VIEWS = ("core", "knowledge", "code", "environment", "context")
STATES = ("approved", "denied", "unchecked")


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _git_head() -> str:
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return run.stdout.strip()


def _file_hash(path: str) -> str | None:
    file = ROOT / path
    return hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None


def _safe(path: str) -> str:
    return atlas.canonical(path)


def _ledger_path(name: str) -> Path:
    atlas.location(name)  # Reuse the existing safe ID grammar.
    return WORKSPACE / name / "review.json"


def _intent(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: plan[key]
        for key in (
            "schema",
            "kind",
            "id",
            "goal",
            "base_sha",
            "source_identities",
            "authority_sha256",
            "items",
            "groups",
            "created_for",
            "default_enabled",
        )
        if key in plan
    } | {
        "items": [
            {key: value for key, value in item.items() if key != "proposal_review"}
            for item in plan["items"]
        ]
    }


def validate(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA or plan.get("kind") != "human_check":
        raise atlas.FlowError("Human Check schema is invalid")
    if plan.get("seal") != _hash(_intent(plan)):
        raise atlas.FlowError("Human Check proposal changed after review creation")
    if plan.get("mode") not in ("on", "off"):
        raise atlas.FlowError("Invalid review mode")
    if type(plan.get("revision")) is not int or plan["revision"] < 0:
        raise atlas.FlowError("Invalid review revision")
    for item in plan["items"]:
        decision = item.get("proposal_review")
        if decision is not None and (
            not isinstance(decision, dict) or decision.get("decision") not in STATES[:2]
        ):
            raise atlas.FlowError("Invalid independent node decision")
    if not isinstance(plan.get("viewed", {}), dict):
        raise atlas.FlowError("Invalid review coverage ledger")
    groups = plan.get("groups", [])
    if not isinstance(groups, list):
        raise atlas.FlowError("Invalid review groups")
    known = {item["key"] for item in plan["items"]}
    grouped = [key for group in groups for key in group["item_keys"]]
    if groups and (set(grouped) != known or len(grouped) != len(set(grouped))):
        raise atlas.FlowError("Every proposed Atlas part must have exactly one review group")
    if plan["mode"] == "off" and not plan.get("opt_out"):
        raise atlas.FlowError("Human Check can only be disabled by human action")
    return plan


def load(plan_id: str) -> dict[str, Any]:
    plan = json.loads(_ledger_path(plan_id).read_text(encoding="utf-8"))
    if plan.get("id") != plan_id:
        raise atlas.FlowError("Human Check plan ID mismatch")
    return validate(plan)


def create(manifest: dict[str, Any], *, base_sha: str | None = None) -> dict[str, Any]:
    """Prepare a proposal from confirmed context without touching source files."""
    plan_id = manifest.get("id")
    destination = _ledger_path(plan_id)
    if destination.exists():
        raise atlas.FlowError("Existing Human Check may not be overwritten")
    goal = manifest.get("goal")
    if not isinstance(goal, str) or len(goal.strip()) < 12:
        raise atlas.FlowError("Human Check needs a concrete goal")
    if "default_enabled" in manifest and manifest["default_enabled"] is not True:
        raise atlas.FlowError("Human Check must begin enabled")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 150:
        raise atlas.FlowError("Human Check requires 1..150 independent changes")
    items, names, sources = [], set(), {}
    context_ids = {
        term["id"]
        for path in ("doc/knowledge/glossary.json", "doc/knowledge/atlas_terms.json")
        for term in json.loads((ROOT / path).read_text(encoding="utf-8"))["terms"]
    }
    for row in raw_items:
        if not isinstance(row, dict):
            raise atlas.FlowError("Invalid change item")
        view, node_id, operation = (row.get("view"), row.get("node_id"), row.get("operation"))
        if view not in VIEWS or operation not in ("add", "change", "remove"):
            raise atlas.FlowError("Invalid Atlas change view or operation")
        if not isinstance(node_id, str) or not node_id or len(node_id) > 220:
            raise atlas.FlowError("Invalid node ID")
        key = view + ":" + node_id
        if key in names:
            raise atlas.FlowError("Repeated Atlas change")
        names.add(key)
        originals = (
            atlas.view_nodes(view)
            if view != "core"
            else {
                node["id"]: node
                for node in json.loads(
                    (ROOT / "harness/core/project_graph.json").read_text(encoding="utf-8")
                )["nodes"]
            }
        )
        original = originals.get(node_id)
        if (operation == "add") == (original is not None):
            raise atlas.FlowError("Atlas node presence conflicts with proposed operation: " + key)
        files = row.get("files")
        if not isinstance(files, list) or not files or len(set(files)) != len(files):
            raise atlas.FlowError("Every Atlas change needs distinct source paths")
        for path in files:
            _safe(path)
            sources[path] = _file_hash(path)
        after = row.get("after")
        if view == "code":
            source_path = (original or {}).get("path") or (after or {}).get("path")
            if (
                not isinstance(source_path, str)
                or not source_path.endswith(".py")
                or source_path not in files
            ):
                raise atlas.FlowError("Code proposal must bind the exact Python source path")
        if operation != "remove" and (
            not isinstance(after, dict)
            or any(
                not isinstance(after.get(k), str) or not after[k].strip()
                for k in ("title", "description")
            )
        ):
            raise atlas.FlowError("Proposed nodes require an explanatory title and description")
        if operation == "remove" and after is not None:
            raise atlas.FlowError("Removed nodes cannot have new values")
        terms = row.get("context_term_ids", [])
        if (
            not isinstance(terms, list)
            or len(set(terms)) != len(terms)
            or (not set(terms) <= context_ids)
        ):
            raise atlas.FlowError("Invalid Context reference")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise atlas.FlowError("Every change needs a Context-grounded reason")
        items.append(
            {
                "key": key,
                "view": view,
                "node_id": node_id,
                "operation": operation,
                "parent_id": row.get("parent_id"),
                "before": atlas.brief(original),
                "after": after,
                "files": files,
                "reason": row["reason"],
                "risk": str(row.get("risk", "")),
                "checks": row.get("checks", []),
                "context_term_ids": terms,
                "proposal_review": None,
            }
        )
    proposed_base = base_sha or _git_head()
    if not re.fullmatch(r"[0-9a-f]{40}", proposed_base):
        raise atlas.FlowError("Invalid source commit SHA")
    # Multi-node review groups are optional for legacy single-node proposals.
    # Each group independently authorizes one change that can touch several
    # Atlas trees; all its source-backed parts must be visited first.
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        entry = next(x for x in raw_items if x["view"] == item["view"] and
                     x["node_id"] == item["node_id"])
        group_id = entry.get("group_id")
        if group_id is None:
            continue
        if not isinstance(group_id, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", group_id) is None:
            raise atlas.FlowError("Invalid Human Check review group ID")
        title = entry.get("group_title")
        if not isinstance(title, str) or not title.strip():
            raise atlas.FlowError("Grouped review needs a human-readable group title")
        group = grouped.setdefault(group_id, {"id": group_id, "title": title.strip(), "item_keys": []})
        if group["title"] != title.strip():
            raise atlas.FlowError("Conflicting titles within a review group")
        group["item_keys"].append(item["key"])
    if grouped and len([x for x in raw_items if x.get("group_id")]) != len(items):
        raise atlas.FlowError("Grouped Human Check cannot mix group and independent nodes")
    plan = {
        "schema": SCHEMA,
        "kind": "human_check",
        "id": plan_id,
        "goal": goal.strip(),
        "created_for": manifest.get("created_for", "GitHub source change"),
        "base_sha": proposed_base,
        "source_identities": sources,
        "authority_sha256": {p: _file_hash(p) for p in AUTHORITIES},
        "default_enabled": True,
        "mode": "on",
        "opt_out": None,
        "revision": 0,
        "items": items,
        "groups": list(grouped.values()),
        "viewed": {},
    }
    plan["seal"] = _hash(_intent(plan))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = report_markdown(plan)
    (destination.parent / "report.md").write_text(report, encoding="utf-8")
    return plan


def report_markdown(plan: dict[str, Any]) -> str:
    validate(plan)
    lines = [
        "# Human Check · " + plan["id"],
        "",
        "Goal: " + plan["goal"],
        "",
        "Baseline: " + plan["base_sha"],
        "",
        "Default: ON; each Atlas item is approved or denied independently.",
        "",
        "| Tree | Node | Operation | Files | Decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    if plan["groups"]:
        lines.extend(["", "## Review groups (one decision after all parts have been visited)", ""])
        for group in plan["groups"]:
            seen = plan["viewed"].get(group["id"], {})
            state = (next((item.get("proposal_review") for item in plan["items"]
                      if item["key"] == group["item_keys"][0]), None) or {}).get("decision", "unchecked")
            lines.append(f"- **{group['id']} · {group['title']}**: {len(seen)}/{len(group['item_keys'])} viewed; {state}")
    for item in plan["items"]:
        status = (item.get("proposal_review") or {}).get("decision", "unchecked")
        lines.append(
            "| "
            + " | ".join(
                str(x).replace("|", "\\|")
                for x in (
                    item["view"],
                    item["node_id"],
                    item["operation"],
                    ", ".join(item["files"]),
                    status,
                )
            )
            + " |"
        )
    for item in plan["items"]:
        lines.extend(
            [
                "",
                "## " + item["key"],
                "",
                "**Why:** " + item["reason"],
                "",
                "**Before:** " + (item["before"] or {}).get("description", "New node"),
                "",
                "**Proposed after:** "
                + ((item["after"] or {}).get("description", "Remove this item")),
                "",
                "**Risk:** " + item["risk"],
                "",
                "**Verification:** " + ", ".join(map(str, item["checks"])),
                "",
                "**Context:** " + ", ".join(item["context_term_ids"]),
                "",
            ]
        )
    return "\n".join(lines)


def snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    """Use the existing Atlas overlay and cards, including read-only Core."""
    from harness.core.agent_proposal_review import selected_context

    # The published Atlas snapshot can lag behind current Core authority.
    # Ship the real source-backed Core hierarchy inside the *local* review
    # response so existing nodes such as q.reproducibility remain reachable.
    core = json.loads((ROOT / "harness/core/project_graph.json").read_text(encoding="utf-8"))
    return {"plan": validate(plan), "context": selected_context(plan), "core_context": core}


def _write(plan: dict[str, Any]) -> None:
    validate(plan)
    destination = _ledger_path(plan["id"])
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(destination)
    (destination.parent / "report.md").write_text(report_markdown(plan), encoding="utf-8")


def visit(plan: dict[str, Any], key: str, *, actor: str) -> dict[str, Any]:
    """Persist a human's view of one actual Atlas proposal part."""
    validate(plan)
    if plan["mode"] == "off" or not actor.strip():
        raise atlas.FlowError("Review must be enabled and human-authenticated")
    for group in plan["groups"]:
        if key in group["item_keys"]:
            plan["viewed"].setdefault(group["id"], {})[key] = {
                "at": atlas.now(), "actor": actor,
            }
            plan["revision"] += 1
            _write(plan)
            return snapshot(plan)
    raise atlas.FlowError("Atlas proposal part is not in a review group")


def decide_group(plan: dict[str, Any], group_id: str, decision: str, *, actor: str) -> dict[str, Any]:
    """Authorize a logical change only after *every* related Atlas part was viewed."""
    validate(plan)
    if plan["mode"] == "off" or not actor.strip() or decision not in STATES:
        raise atlas.FlowError("Invalid authenticated group decision")
    group = next((g for g in plan["groups"] if g["id"] == group_id), None)
    if group is None:
        raise atlas.FlowError("Unknown Human Check review group")
    seen = plan["viewed"].get(group_id, {})
    if decision != "unchecked" and not set(group["item_keys"]) <= set(seen):
        raise atlas.FlowError("Visit every linked Atlas part before approving or denying this group")
    for item in plan["items"]:
        if item["key"] not in group["item_keys"]:
            continue
        item["proposal_review"] = None if decision == "unchecked" else {
            "decision": decision, "actor": actor, "at": atlas.now(),
            "group_id": group_id,
        }
    plan["revision"] += 1
    _write(plan)
    return snapshot(plan)


def decide(plan: dict[str, Any], key: str, decision: str, *, actor: str) -> dict[str, Any]:
    """Local authenticated Atlas decides one item; never writes authority graphs."""
    validate(plan)
    if plan["mode"] == "off":
        raise atlas.FlowError("Human Check disabled; decisions unavailable")
    if decision not in STATES:
        raise atlas.FlowError("Unknown review decision")
    item = next((row for row in plan["items"] if row["key"] == key), None)
    if item is None:
        raise atlas.FlowError("Unknown independent change")
    if not actor.strip():
        raise atlas.FlowError("Human reviewer identity required")
    if any(key in group["item_keys"] for group in plan.get("groups", [])):
        raise atlas.FlowError("Grouped Atlas part cannot bypass review-group decision")
    item["proposal_review"] = (
        None
        if decision == "unchecked"
        else {
            "decision": decision,
            "actor": actor,
            "at": atlas.now(),
        }
    )
    plan["revision"] += 1
    _write(plan)
    return snapshot(plan)


def set_mode(plan: dict[str, Any], enabled: bool, *, actor: str) -> dict[str, Any]:
    """Explicit authenticated opt-out; no implicit agent-side bypass."""
    validate(plan)
    if type(enabled) is not bool or not actor.strip():
        raise atlas.FlowError("Human Check toggle requires explicit human decision")
    plan["mode"] = "on" if enabled else "off"
    plan["opt_out"] = None if enabled else {"actor": actor, "at": atlas.now()}
    plan["revision"] += 1
    _write(plan)
    return snapshot(plan)


def allowed_paths(
    plan: dict[str, Any],
    paths: list[str],
    *,
    base_sha: str,
    before_edit: bool = True,
) -> dict[str, Any]:
    """Reject unapproved or mixed-decision file scopes; keep independent successes."""
    validate(plan)
    if base_sha != plan["base_sha"]:
        raise atlas.FlowError("Source commit drift; refresh the Human Check")
    if not paths or len(set(paths)) != len(paths):
        raise atlas.FlowError("Nonempty distinct path scope required")
    if before_edit:
        for authority, expected in plan["authority_sha256"].items():
            if _file_hash(authority) != expected:
                raise atlas.FlowError("Atlas authority drift; refresh proposal: " + authority)
    selected = []
    for path in paths:
        _safe(path)
        matches = [row for row in plan["items"] if path in row["files"]]
        if plan["mode"] == "on" and (
            not matches
            or any(
                (row.get("proposal_review") or {}).get("decision") != "approved" for row in matches
            )
        ):
            raise atlas.FlowError("Unapproved or overlapping denied scope: " + path)
        if before_edit and _file_hash(path) != plan["source_identities"].get(path):
            raise atlas.FlowError("Source changed before approved implementation: " + path)
        selected.append(path)
    return {
        "status": "HUMAN_CHECK_OPTED_OUT" if plan["mode"] == "off" else "APPROVED_SCOPE",
        "paths": selected,
        "base_sha": plan["base_sha"],
        "approved_items": [
            row["key"]
            for row in plan["items"]
            if (row.get("proposal_review") or {}).get("decision") == "approved"
            and any(p in selected for p in row["files"])
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("prepare")
    create_parser.add_argument("--manifest", type=Path, required=True)
    create_parser.add_argument("--base-sha")
    for action in ("report", "inspect", "serve", "check"):
        parser = sub.add_parser(action)
        parser.add_argument("id")
        if action == "serve":
            parser.add_argument("--port", type=int, default=8766)
        if action == "check":
            parser.add_argument("--base-sha", required=True)
            parser.add_argument("--after-edit", action="store_true")
            parser.add_argument("--path", action="append", required=True)
    args = p.parse_args()
    try:
        if args.command == "prepare":
            data = json.loads(args.manifest.read_text(encoding="utf-8"))
            result = create(data, base_sha=args.base_sha)
            print(
                json.dumps(
                    {
                        "id": result["id"],
                        "report": str(_ledger_path(result["id"]).parent / "report.md"),
                    }
                )
            )
        else:
            plan = load(args.id)
            if args.command == "inspect":
                print(json.dumps(plan, indent=2, ensure_ascii=False))
            elif args.command == "report":
                print(report_markdown(plan))
            elif args.command == "check":
                if not args.after_edit and _git_head() != args.base_sha:
                    raise atlas.FlowError("Current HEAD differs from approved source commit")
                print(
                    json.dumps(
                        allowed_paths(
                            plan,
                            args.path,
                            base_sha=args.base_sha,
                            before_edit=not args.after_edit,
                        ),
                        indent=2,
                    )
                )
            else:
                from harness.core.agent_review_server import serve

                serve(args.id, args.port, human_check=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        p.exit(2, "HUMAN CHECK BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
