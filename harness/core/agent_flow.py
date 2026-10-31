"""Auditable agent execution plan. Only REVIEW DATA is stored here; no model runs."""

from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "harness/github/work_plan/agent_flow"
VIEWS = ("knowledge", "code", "environment", "context")
SCHEMA = "openresearch.agent_flow.v1"
PROTECTED = (".cursor/", ".github/", "harness/", "saebase/runtime/")
EDITABLE = (
    "datasets/",
    "models/",
    "tasks/",
    "conf/",
    "saebase/",
    "doc/knowledge/",
    "doc/code/",
    "tests/",
    "experiments/",
    "scripts/",
    "saebase/tools/",
)
AUTHORITY = {
    "knowledge": "doc/knowledge/context_graph.json",
    "code": "doc/code/code_tree.json",
    "context": "doc/knowledge/glossary.json",
    "environment": "harness/environment/example.json",
}


class FlowError(ValueError):
    pass


def need(test, msg):
    if not test:
        raise FlowError(msg)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical(path, exists=False):
    need(
        isinstance(path, str)
        and path
        and path.strip() == path
        and not path.startswith("/")
        and "\\" not in path
        and all(p not in ("", ".", "..") for p in path.split("/"))
        and PurePosixPath(path).as_posix() == path,
        "Unsafe relative path",
    )
    absolute = (ROOT / path).resolve(strict=False)
    need(absolute.is_relative_to(ROOT.resolve()), "Symlink escape forbidden")
    if exists:
        need(absolute.is_file(), "Missing source or evidence: " + path)
    return path


def editable(path):
    canonical(path)
    need(
        any(path.startswith(x) for x in EDITABLE)
        and not any(path.startswith(x) for x in PROTECTED),
        "Locked or unlisted path: " + path,
    )
    need(path != "doc/code/code_tree.json", "Code Tree is derived from AST; edit source instead")
    need(not path.endswith(("/", ".pyc")), "Expected a human-audited source file")
    return path


def checksum(path):
    return hashlib.sha256((ROOT / canonical(path, True)).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def location(plan_id):
    need(
        isinstance(plan_id, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", plan_id) is not None,
        "Invalid plan identifier",
    )
    return PLANS / (plan_id + ".json")


def immutable_snapshot(obj):
    """Fields human reviewers actually authorized; mutable evidence is excluded."""
    return {
        "id": obj["id"],
        "goal": obj["goal"],
        "scope": obj["allowed_paths"],
        "source": obj["source_plan"],
        "authorized_commands": obj["authorized_commands"],
        "authority": obj["authority_digest"],
        "atlas_terms_sha256": obj["atlas_terms_sha256"],
        "file_sha256_before": obj["file_sha256_before"],
        "items": [
            {
                key: item.get(key)
                for key in (
                    "key",
                    "view",
                    "node_id",
                    "operation",
                    "before",
                    "after",
                    "parent_id",
                    "files",
                    "before_ast_sha256",
                )
            }
            for item in obj["items"]
        ],
    }


def validate_seal(obj):
    need(obj.get("schema") == SCHEMA, "Bad workflow schema")
    need(
        obj.get("preview_sha256") == digest(immutable_snapshot(obj)),
        "Plan intent, source identity, node definitions or scope changed after audit",
    )
    source = obj.get("source_plan")
    if source:
        need(
            checksum(source["path"]) == source["sha256"],
            "Source GitHub work plan changed; a fresh audit is required",
        )
    return obj


def load(plan_id):
    obj = read(location(plan_id))
    need(obj.get("id") == plan_id, "Wrong plan ledger")
    return validate_seal(obj)


def view_nodes(view):
    if view == "environment":
        return {
            "env:" + x: {
                "id": "env:" + x,
                "title": x.capitalize(),
                "description": "Requires real local machine observation",
            }
            for x in ("platform", "cpu", "gpu", "memory", "storage", "execution")
        }
    if view == "context":
        records = []
        for src in ("doc/knowledge/glossary.json", "doc/knowledge/atlas_terms.json"):
            records += read(ROOT / src).get("terms", [])
        return {
            r["id"]: {"id": r["id"], "title": r["term"], "description": r["definition"]}
            for r in records
        }
    return {r["id"]: r for r in read(ROOT / AUTHORITY[view])["nodes"]}


def code_ast_fingerprint(node_id: str, path: str | None) -> str | None:
    """Fingerprint the actual named AST node, ignoring line-number changes."""
    if not path or not path.endswith(".py"):
        return None
    canonical(path)
    source = ROOT / path
    if not source.is_file():
        return None
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=path)
    segments = node_id.split(":")
    kind = segments[1] if len(segments) > 1 else None
    if kind == "module":
        target = tree
    elif kind in ("function", "class", "method") and len(segments) == 4:
        name = segments[3]
        if kind == "method":
            if "." not in name:
                return None
            cls_name, method_name = name.split(".", 1)
            cls = next(
                (x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls_name), None
            )
            target = (
                next(
                    (
                        x
                        for x in cls.body
                        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and x.name == method_name
                    ),
                    None,
                )
                if cls
                else None
            )
        elif kind == "class":
            target = next(
                (x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name), None
            )
        else:
            target = next(
                (
                    x
                    for x in tree.body
                    if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name
                ),
                None,
            )
    else:
        return None
    if target is None:
        return None
    return hashlib.sha256(ast.dump(target, include_attributes=False).encode()).hexdigest()


def brief(node):
    if node is None:
        return None
    return {
        k: node[k]
        for k in ("id", "title", "description", "signature", "status", "path", "line")
        if k in node
    }


def intake(manifest, source=None):
    need(manifest.get("version") == 1, "Manifest requires version 1")
    plan_id = manifest.get("id")
    location(plan_id)
    goal = manifest.get("goal")
    need(isinstance(goal, str) and len(goal.strip()) >= 12, "Plan needs a concrete goal")
    files = manifest.get("allowed_paths")
    need(isinstance(files, list) and 1 <= len(files) <= 150, "Explicit allowed_paths required")
    files = [editable(v) for v in files]
    need(len(files) == len(set(files)), "Repeated execution path")
    commands = manifest.get("authorized_commands", [])
    need(
        isinstance(commands, list)
        and len(commands) <= 40
        and all(
            isinstance(cmd, str)
            and 2 <= len(cmd) <= 500
            and not any(x in cmd for x in (";", "&&", "||", "|", ">", "<", "`", "$(", "\\n"))
            for cmd in commands
        ),
        "Commands must be explicit single non-shell commands",
    )
    for cmd in commands:
        try:
            args = shlex.split(cmd)
        except ValueError as exc:
            raise FlowError("Invalid authorized command") from exc
        need(
            args
            and not any(
                forbidden in cmd
                for forbidden in ("harness/core/", "harness/visualization/", ".cursor/", ".github/")
            ),
            "Approved scientific command must never target governance",
        )
        need(
            args[0] not in ("sh", "bash", "zsh", "pwsh", "powershell", "sudo")
            and not (
                args[0] in ("python", "python3", "node")
                and any(flag in args for flag in ("-c", "-e", "--eval"))
            ),
            "Shell interpreters and arbitrary inline code need a separately governed execution environment",
        )
    need(len(set(commands)) == len(commands), "Duplicate approved execution command")
    changes = manifest.get("changes")
    need(
        isinstance(changes, list) and 1 <= len(changes) <= 300,
        "1..300 explicit tree edits required",
    )
    origin = None
    if source is not None:
        canonical(source, True)
        need(
            source.startswith("harness/github/work_plan/"),
            "Only external GitHub work plan sources are accepted",
        )
        txt = (ROOT / source).read_text(encoding="utf-8")
        need(
            not re.search(r"(?im)^\s*(status:\s*DEFERRED|execution:\s*NOT_AUTHORIZED)\s*$", txt),
            "Deferred work plan is not an authorization",
        )
        origin = {"path": source, "sha256": checksum(source)}
    nodes, items, seen = {}, [], set()
    for change in changes:
        need(isinstance(change, dict), "Expected a node-change object")
        view, action, node = (change.get(k) for k in ("view", "operation", "node_id"))
        need(
            view in VIEWS and action in ("add", "change", "remove"),
            "Core editing and unsupported operations are forbidden",
        )
        need(isinstance(node, str) and 1 <= len(node) <= 220, "Stable node_id required")
        key = view + ":" + node
        need(key not in seen, "Duplicate change: " + key)
        seen.add(key)
        if view not in nodes:
            nodes[view] = view_nodes(view)
        original = nodes[view].get(node)
        need(
            original is None if action == "add" else original is not None,
            "Node presence does not match operation: " + key,
        )
        after = change.get("after")
        if action != "remove":
            need(
                isinstance(after, dict)
                and all(
                    isinstance(after.get(k), str) and after[k].strip()
                    for k in ("title", "description")
                ),
                "Changed node requires title and description",
            )
        else:
            need(after is None, "Remove action must omit 'after'")
        if view == "code":
            need(
                node.startswith(("code:module:", "code:class:", "code:function:", "code:method:")),
                "Only source-backed Code nodes can be included in an execution plan",
            )
            source_path = (original or {}).get("path") or (after or {}).get("path")
            need(
                isinstance(source_path, str) and source_path.endswith(".py"),
                "Code node needs a source file path",
            )
        item_files = change.get("files")
        need(
            isinstance(item_files, list) and item_files and all(p in files for p in item_files),
            "Node files must be in audited scope",
        )
        if view == "code":
            need(
                source_path in item_files,
                "Code node must identify its own source in its approved file list",
            )
        items.append(
            {
                "key": key,
                "view": view,
                "node_id": node,
                "operation": action,
                "before": brief(original),
                "after": after,
                "parent_id": change.get("parent_id"),
                "files": item_files,
                "before_ast_sha256": (
                    code_ast_fingerprint(node, original.get("path"))
                    if view == "code" and original
                    else None
                ),
                "state": "not_implemented",
                "decision": None,
                "reviewer": None,
                "marks": [],
                "evidence": [],
            }
        )
    refs = {view: checksum(AUTHORITY[view]) for view in nodes}
    obj = {
        "schema": SCHEMA,
        "id": plan_id,
        "goal": goal.strip(),
        "source_plan": origin,
        "allowed_paths": files,
        "authorized_commands": commands,
        "items": items,
        "authority_digest": refs,
        "atlas_terms_sha256": checksum("doc/knowledge/atlas_terms.json"),
        "file_sha256_before": {p: checksum(p) if (ROOT / p).is_file() else None for p in files},
        "created_at": now(),
        "revision": 0,
        "phase": "preflight",
        "authorization": None,
        "implementation": None,
        "observation": None,
        "observations": [],
        "events": [{"event": "intake", "at": now()}],
        "deployment": [
            "Audit source, constraints, node IDs and candidate changes",
            "Show proposed RED overlay and human-readable before/after text",
            "Obtain direct human preflight authorization of bounded files",
            "Implement task, review source changes, record YELLOW overlay",
            "Execute approved action and audit real result independently",
            "Human marks content changes and chooses GREEN/YELLOW/RED/DELETE",
            "Finalize with evidence hashes, decisions and next actions",
        ],
    }
    obj["preview_sha256"] = digest(immutable_snapshot(obj))
    return obj


def save(obj, action, **detail):
    validate_seal(obj)
    obj["revision"] += 1
    obj["events"].append({"event": action, "at": now(), **detail})
    write(location(obj["id"]), obj)


def authorize(obj, actor, *, human=False, channel="local_review"):
    validate_seal(obj)
    need(human, "Only the interactive human review channel can authorize")
    need(channel in ("local_review", "github_pr_review"), "Invalid authenticated review channel")
    need(obj["phase"] == "preflight", "Wrong workflow phase")
    need(isinstance(actor, str) and len(actor.strip()) >= 2, "Human identity required")
    for path, original in obj["file_sha256_before"].items():
        current = checksum(path) if (ROOT / path).is_file() else None
        need(current == original, "Source changed since preflight: " + path)
    obj["authorization"] = {
        "actor": actor.strip(),
        "at": now(),
        "channel": channel,
        "preview_sha256": obj["preview_sha256"],
        "scope": obj["allowed_paths"],
        "commands": obj["authorized_commands"],
    }
    obj["phase"] = "authorized"
    save(obj, "human_authorized", actor=actor.strip())


def scope(obj, paths):
    validate_seal(obj)
    need(
        obj["phase"] == "authorized",
        "Source writes freeze after code review; open a new plan to reimplement",
    )
    auth = obj.get("authorization")
    need(
        auth and auth.get("preview_sha256") == obj["preview_sha256"],
        "Preflight has not received direct human approval",
    )
    for path in paths:
        need(editable(path) in auth["scope"], "Out-of-scope edit: " + path)
    need(bool(paths), "No changed paths supplied")


def stage(obj, files, evidence):
    validate_seal(obj)
    need(obj["phase"] == "authorized", "Authorize preflight before editing")
    scope(obj, files)
    need(bool(evidence), "Implementation needs code review evidence")
    initial = obj.get("file_sha256_before", {})
    actual = {path: checksum(path) for path in files}
    need(
        all(actual[path] != initial.get(path) for path in files),
        "Stage requires observed source edits, not only a claimed plan",
    )
    proofs = [{"path": canonical(x, True), "sha256": checksum(x)} for x in evidence]
    obj["implementation"] = {
        "review_evidence": proofs,
        "files": [{"path": canonical(x, True), "sha256": checksum(x)} for x in files],
        "status": "reviewed_unverified",
        "recorded_at": now(),
    }
    obj["phase"] = "partial"
    for item in obj["items"]:
        if any(path in item["files"] for path in files):
            item["state"] = "partial"
            item["evidence"] += proofs
    save(obj, "code_reviewed")


def observe(obj, result, evidence, nodes):
    validate_seal(obj)
    need(obj["phase"] in ("partial", "review"), "Implementation review must precede execution")
    need(bool(evidence), "Actual result needs independent evidence")
    eligible = {x["key"] for x in obj["items"] if x["state"] == "partial"}
    need(
        isinstance(nodes, list)
        and bool(nodes)
        and len(nodes) == len(set(nodes))
        and set(nodes) <= eligible,
        "Explicit partial node keys must be mapped to actual result evidence",
    )
    canonical(result, True)
    data = (ROOT / result).read_bytes()
    need(len(data) <= 10_000_000, "Large result must be summarized first")
    if result.endswith(".json"):
        value = json.loads(data)
        need(
            isinstance(value, dict)
            and value.get("synthetic") is not True
            and value.get("status") not in ("configured_only", "planned", "not_run"),
            "Synthetic or configured-only result cannot establish execution",
        )
    observation = {
        "result": {"path": result, "sha256": checksum(result)},
        "evidence": [{"path": canonical(x, True), "sha256": checksum(x)} for x in evidence],
        "status": "requires_human_scientific_audit",
        "recorded_at": now(),
        "node_keys": nodes,
    }
    obj.setdefault("observations", []).append(observation)
    obj["observation"] = observation  # Last result, for old clients only.
    obj["phase"] = "review"
    save(obj, "observed_result", nodes=nodes, result_sha256=observation["result"]["sha256"])


def decide(obj, key, action, actor, marks=None, reason="", *, human=False, channel="local_review"):
    validate_seal(obj)
    need(human, "Human-only decision; agents may submit proposals, not approval")
    need(channel in ("local_review", "github_pr_review"), "Invalid authenticated review channel")
    need(obj["phase"] in ("partial", "review"), "Stage implementation first")
    need(action in ("approve", "keep_partial", "retry", "delete"), "Invalid disposition")
    matches = [x for x in obj["items"] if x["key"] == key]
    need(len(matches) == 1, "Unknown reviewed node")
    item = matches[0]
    need(item["decision"] is None, "Node has already been reviewed; use a fresh follow-up plan")
    if action == "approve":
        need(
            obj["phase"] == "review"
            and obj.get("observation") is not None
            and any(key in x["node_keys"] for x in obj.get("observations", []))
            and item["state"] == "partial",
            "GREEN needs node-specific results, partial implementation and human review",
        )
        need(len(reason.strip()) >= 8, "Green review requires specific reasoning")
    if action == "keep_partial":
        need(item["state"] == "partial", "No implemented work to retain as partial")
    need(isinstance(actor, str) and len(actor.strip()) >= 2, "Reviewer identity required")
    marks = marks or []
    need(
        isinstance(marks, list)
        and len(marks) <= 25
        and all(isinstance(m, str) and 0 < len(m) <= 300 for m in marks),
        "Marks are small human-selected text fragments",
    )
    proposed_text = " ".join(
        str(v) for v in (*(item.get("after") or {}).values(), *(item.get("before") or {}).values())
    )
    need(all(m in proposed_text for m in marks), "Human mark is not in proposed content")
    item["marks"] = marks
    item["decision"] = action
    item["state"] = {
        "approve": "implemented",
        "keep_partial": "partial",
        "retry": "not_implemented",
        "delete": "deleted",
    }[action]
    item["reviewer"] = {
        "actor": actor.strip(),
        "at": now(),
        "channel": channel,
        "reason": reason.strip(),
    }
    save(obj, "human_decision", key=key, disposition=action, actor=actor.strip())


def check_evidence_sealed(obj):
    """Recheck all staged source, review evidence and result bytes at finalization."""
    validate_seal(obj)
    implementation = obj.get("implementation")
    need(
        implementation and implementation.get("files"), "Reviewed source modifications are required"
    )
    for item in implementation["files"] + implementation.get("review_evidence", []):
        need(
            checksum(item["path"]) == item["sha256"],
            "Reviewed code or review proof changed: " + item["path"],
        )
    for observation in obj.get("observations", []):
        for item in [observation["result"], *observation.get("evidence", [])]:
            need(
                checksum(item["path"]) == item["sha256"],
                "Result or audit proof changed after observation: " + item["path"],
            )


def finalize(obj):
    check_evidence_sealed(obj)
    need(obj["phase"] in ("partial", "review"), "Wrong finalize phase")
    need(
        all(x["decision"] for x in obj["items"]),
        "Every node must have an explicit human disposition",
    )
    from harness.core.agent_apply import apply as adopt

    adopted_paths = adopt(obj)
    report = {
        "schema": "openresearch.agent_flow.final.v1",
        "id": obj["id"],
        "adopted_authority_paths": adopted_paths,
        "preview_sha256": obj["preview_sha256"],
        "authorized_scope": obj["authorization"],
        "implementation": obj["implementation"],
        "observation": obj["observation"],
        "observations": obj.get("observations", []),
        "changes": [
            {
                "key": x["key"],
                "view": x["view"],
                "operation": x["operation"],
                "before": x["before"],
                "after": x["after"],
                "status": x["state"],
                "decision": x["decision"],
                "marks": x["marks"],
                "reviewer": x["reviewer"],
                "evidence": x["evidence"],
                "result_artifacts": [
                    r["result"] for r in obj.get("observations", []) if x["key"] in r["node_keys"]
                ],
            }
            for x in obj["items"]
        ],
        "caveat": "Code is sourced from actual AST and research results need separate audit. "
        "Knowledge/Context authority updates require validated application.",
    }
    obj["phase"] = "finalized"
    obj["adopted_authority_paths"] = adopted_paths
    save(obj, "finalized", adopted=adopted_paths)
    write(PLANS / (obj["id"] + ".final.json"), report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    a = s.add_parser("intake")
    a.add_argument("manifest")
    a.add_argument("--source")
    a = s.add_parser("inspect")
    a.add_argument("plan")
    a = s.add_parser("browse", help="Select exact existing non-Core tree nodes")
    a.add_argument("view", choices=VIEWS)
    a.add_argument("node_ids", nargs="+")
    a = s.add_parser("preview", help="Concise change list from audited plan")
    a.add_argument("plan")
    a = s.add_parser("scope")
    a.add_argument("plan")
    a.add_argument("paths", nargs="+")
    a = s.add_parser("stage")
    a.add_argument("plan")
    a.add_argument("--changed", action="append", required=True)
    a.add_argument("--evidence", action="append", required=True)
    a = s.add_parser("observe")
    a.add_argument("plan")
    a.add_argument("--result", required=True)
    a.add_argument("--evidence", action="append", required=True)
    a.add_argument("--node", action="append", required=True, help="covered partial node key")
    a = s.add_parser("finalize")
    a.add_argument("plan")
    a = s.add_parser("serve")
    a.add_argument("plan")
    a.add_argument("--port", type=int, default=8766)
    x = p.parse_args()
    try:
        if x.command == "intake":
            manifest_path = canonical(x.manifest, True)
            obj = intake(read(ROOT / manifest_path), x.source or manifest_path)
            need(not location(obj["id"]).exists(), "Plan exists; no silent overwrite")
            write(location(obj["id"]), obj)
            print(
                json.dumps(
                    {
                        "plan": obj["id"],
                        "phase": obj["phase"],
                        "preview_sha256": obj["preview_sha256"],
                        "count": len(obj["items"]),
                    }
                )
            )
        elif x.command == "serve":
            from harness.core.agent_review_server import serve

            serve(x.plan, x.port)
        elif x.command == "browse":
            records = view_nodes(x.view)
            need(
                len(x.node_ids) <= 50 and len(set(x.node_ids)) == len(x.node_ids),
                "Select no more than 50 distinct node IDs",
            )
            need(all(key in records for key in x.node_ids), "Unknown node in selected view")
            print(
                json.dumps(
                    {
                        "view": x.view,
                        "nodes": [brief(records[key]) for key in x.node_ids],
                        "authority": AUTHORITY[x.view],
                        "core_excluded": True,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            obj = load(x.plan)
            if x.command == "inspect":
                print(json.dumps(obj, ensure_ascii=False, indent=2))
            elif x.command == "preview":
                print(
                    json.dumps(
                        {
                            "id": obj["id"],
                            "phase": obj["phase"],
                            "preview_sha256": obj["preview_sha256"],
                            "deployment": obj["deployment"],
                            "items": [
                                {
                                    "key": item["key"],
                                    "state": item["state"],
                                    "title": (item.get("after") or item.get("before") or {}).get(
                                        "title"
                                    ),
                                    "decision": item["decision"],
                                }
                                for item in obj["items"]
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            elif x.command == "scope":
                scope(obj, x.paths)
                print("AUTHORIZED execution scope")
            elif x.command == "stage":
                stage(obj, x.changed, x.evidence)
                print("PARTIAL")
            elif x.command == "observe":
                observe(obj, x.result, x.evidence, x.node)
                print("READY FOR HUMAN REVIEW")
            else:
                report = finalize(obj)
                print(json.dumps({"status": "finalized", "changes": len(report["changes"])}))
    except (FlowError, ValueError, KeyError, OSError, TypeError) as exc:
        print("AGENT FLOW ERROR:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
