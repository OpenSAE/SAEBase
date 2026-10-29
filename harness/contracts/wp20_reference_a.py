"""Independent reference X01-X05: actual AST, graph, state, evidence and filesystem."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique(values):
    if len(set(values)) != len(values):
        raise ValueError("duplicate id")


def x01(p, root):
    if set(p) != {"before", "after"}:
        raise ValueError("schema")
    def parse(src):
        if not isinstance(src, str):
            raise ValueError("source")
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            raise ValueError("Invalid Python source") from exc
        return {n.name: {"signature": ast.unparse(n.args), "line": n.lineno}
                for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    before, after = parse(p["before"]), parse(p["after"])
    return {"before": before, "after": after,
            "diff": {"added": sorted(after.keys() - before.keys()),
                     "removed": sorted(before.keys() - after.keys()),
                     "changed": sorted(k for k in before.keys() & after.keys()
                                       if before[k]["signature"] != after[k]["signature"])},
            "evidence": {"before_sha256": digest(p["before"]),
                         "after_sha256": digest(p["after"]), "review": "unchecked"}}


def x02(p, root):
    if set(p) != {"nodes", "terms", "update"}:
        raise ValueError("schema")
    unique([r["id"] for r in p["nodes"]])
    unique([r["id"] for r in p["terms"]])
    nodes = {r["id"]: r for r in p["nodes"]}
    terms = {r["id"]: r for r in p["terms"]}
    update = p["update"]
    tid, nid = update["term"], update["node"]
    if tid not in terms or nid not in nodes:
        raise ValueError("unknown term or node")
    for row in nodes.values():
        for term in row["terms"]:
            if term not in terms or row["id"] not in terms[term]["nodes"]:
                raise ValueError("reverse mismatch")
    for row in terms.values():
        for node in row["nodes"]:
            if node not in nodes or row["id"] not in nodes[node]["terms"]:
                raise ValueError("forward mismatch")
    for row in terms.values():
        if any(not isinstance(row[lang], str) or not row[lang].strip()
               or "unavailable" in row[lang].lower() or "翻译不可用" in row[lang]
               for lang in ("zh", "en")):
            raise ValueError("translation")
    return {"node_ids": sorted(nodes), "term_ids": sorted(terms),
            "links": sorted(f"{nid}:{term}" for nid, r in nodes.items() for term in r["terms"]),
            "updated": {"term": tid, "node": nid, "zh": terms[tid]["zh"], "en": terms[tid]["en"]}}


def x03(p, root):
    if set(p) != {"trees", "events"}:
        raise ValueError("schema")
    trees = {k: {"focus": None, "expanded": [], "decisions": {}, "revision": 0} for k in p["trees"]}
    if set(trees) != {"core", "knowledge", "code", "environment", "context"}:
        raise ValueError("tree set")
    active, lang, history = "core", "en", []
    for event in p["events"]:
        op, tree = event["op"], event.get("tree", active)
        if tree not in trees:
            raise ValueError("tree")
        if op == "switch":
            active = tree
        elif op == "language":
            if event["value"] not in ("zh", "en"):
                raise ValueError("language")
            lang = event["value"]
        elif op in ("focus", "toggle"):
            if event["id"] not in p["trees"][tree]:
                raise ValueError("unknown node")
            if op == "focus":
                trees[tree]["focus"] = event["id"]
            else:
                expanded = trees[tree]["expanded"]
                if event["id"] in expanded:
                    expanded.remove(event["id"])
                else:
                    expanded.append(event["id"])
                    expanded.sort()
        elif op == "review":
            state = trees[tree]
            if (event["id"] not in p["trees"][tree] or event["revision"] != state["revision"]
                    or event["decision"] not in ("approve", "deny", "unchecked")):
                raise ValueError("review")
            state["decisions"][event["id"]] = event["decision"]
            state["revision"] += 1
        else:
            raise ValueError("operation")
        history.append({"op": op, "active": active, "language": lang,
                        "revision": trees[tree]["revision"]})
    return {"active": active, "language": lang, "trees": trees,
            "event_count": len(history), "events": history}


def x04(p, root):
    if set(p) != {"observed", "receipt", "files"}:
        raise ValueError("schema")
    observed = {r["id"]: r for r in p["observed"]}
    unique([r["id"] for r in p["observed"]])
    unique([r["id"] for r in p["receipt"]])
    if set(observed) != {r["id"] for r in p["receipt"]}:
        raise ValueError("execution evidence")
    for r in p["receipt"]:
        o = observed[r["id"]]
        if r["exit"] != o["exit"] or r["command"] != o["command"] or o["exit"] != 0:
            raise ValueError("false receipt")
    if any("/" in r["name"] or r["name"].startswith(".") or r["sha"] != digest(r["content"])
           for r in p["files"]):
        raise ValueError("provenance")
    if any("secret" in str(r).lower() or "token=" in str(r).lower() for r in p["receipt"]):
        raise ValueError("sensitive evidence")
    return {"verified_commands": sorted(observed),
            "source_hashes": {r["name"]: r["sha"] for r in p["files"]},
            "failed_count": 0, "human_approved": False}


def x05(p, root):
    if set(p) != {"allowed", "operations"} or p["allowed"] != "workspace":
        raise ValueError("schema")
    workspace = Path(root).resolve()
    planned = []
    for op in p["operations"]:
        rel = op.get("path")
        if (not isinstance(rel, str) or not rel or rel.startswith(("/", "~"))
                or "\\" in rel or ":" in rel or any(x in ("", ".", "..") for x in rel.split("/"))
                or op.get("op") != "write" or not isinstance(op.get("text"), str)):
            raise ValueError("path or operation")
        target = (workspace / rel).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError("escape")
        planned.append((rel, target, op["text"]))
    results = []
    for rel, target, text in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        results.append({"path": rel, "sha256": digest(text)})
    return {"written": results, "files": sorted(str(x.relative_to(workspace))
            for x in workspace.rglob("*") if x.is_file()), "count": len(results), "scope": "workspace"}


CASES = {"X01": x01, "X02": x02, "X03": x03, "X04": x04, "X05": x05}
