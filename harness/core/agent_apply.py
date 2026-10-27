"""Validate and adopt human-green Knowledge/Context changes atomically.

Code is regenerated from AST, never invented by a plan. Environment is
local-only evidence. Core and Harness governance are NEVER edited.
"""
from __future__ import annotations
import copy
from pathlib import Path
from harness.core import agent_flow as flow
from harness.knowledge.status import refresh_status
from harness.knowledge.validate import validate_scientific
from harness.visualization.export import validate_atlas_vocabulary

KNOWLEDGE = "doc/knowledge/context_graph.json"
GLOSSARY = "doc/knowledge/glossary.json"
VOCAB = "doc/knowledge/atlas_terms.json"
CONTENT_FIELDS = {"title", "description", "term_ids", "evidence",
                  "summary", "type", "status"}
TERM_FIELDS = {"term", "definition", "category", "aliases", "why_it_matters",
               "source_evidence", "foundation", "related_node_ids"}


def result_records(plan: dict, node_key: str) -> list[dict]:
    """All real observations covering one specific proposed change."""
    found = [x for x in plan.get("observations", [])
             if node_key in x.get("node_keys", [])]
    flow.need(bool(found), "No node-specific observations: " + node_key)
    return found


def validated_updates(plan: dict) -> dict[str, dict]:
    approved = [item for item in plan["items"] if item["decision"] == "approve"]
    editing = [item for item in approved if item["view"] in ("knowledge", "context")]
    files = {KNOWLEDGE, GLOSSARY, VOCAB}
    originals = {p: flow.read(flow.ROOT / p) for p in files}
    candidates = copy.deepcopy(originals)
    if editing:
        for path in {KNOWLEDGE if item["view"] == "knowledge" else
                     (GLOSSARY if item["node_id"] in
                      {x["id"] for x in originals[GLOSSARY]["terms"]} else VOCAB)
                     for item in editing}:
            flow.need(plan["authority_digest"].get("knowledge" if path == KNOWLEDGE else "context") is not None,
                      "Missing original authority digest")
            if path == VOCAB:
                flow.need(flow.checksum(path) == plan["atlas_terms_sha256"],
                          "Technical Context changed since preview")
            if path == KNOWLEDGE:
                flow.need(flow.checksum(path) == plan["authority_digest"]["knowledge"],
                          "Scientific graph changed since human preview")
            if path == GLOSSARY:
                flow.need(flow.checksum(path) == plan["authority_digest"]["context"],
                          "Context definitions changed since preview")
    for item in editing:
        path = KNOWLEDGE if item["view"] == "knowledge" else (
            GLOSSARY if any(x["id"] == item["node_id"] for x in
                            originals[GLOSSARY]["terms"]) else VOCAB)
        doc = candidates[path]
        collection = "nodes" if path == KNOWLEDGE else "terms"
        records = doc[collection]
        match = next((x for x in records if x["id"] == item["node_id"]), None)
        action = item["operation"]
        after = item["after"] or {}
        if action == "add":
            flow.need(match is None, "Added node already exists")
            if path == KNOWLEDGE:
                parent = item.get("parent_id")
                flow.need(isinstance(parent, str) and
                          any(n["id"] == parent for n in records),
                          "New knowledge node needs a valid parent")
                flow.need("type" in after and "term_ids" in after and
                          "evidence" in after, "New scientific node needs full typed evidence")
                kind = after["type"]
                parent_kind = next(n["type"] for n in records if n["id"] == parent)
                edge = ("refines" if (parent_kind,kind)==("question","question")
                        else "addresses" if (parent_kind,kind)==("question","solve")
                        else "decomposes")
                doc["edges"].append({"source":parent,"target":item["node_id"],"type":edge})
                record = {k:v for k,v in after.items() if k in CONTENT_FIELDS}
                record.update({"id":item["node_id"],"status":"implemented"})
            else:
                required = TERM_FIELDS - {"term", "definition"}
                flow.need(required <= after.keys(),
                          "New Context term needs complete citations, category and links")
                record = {k:after[k] for k in required}
                record.update({"id": item["node_id"], "term": after["title"],
                               "definition": after["description"]})
            records.append(record)
            continue
        flow.need(match is not None, "Changed source node no longer exists")
        if action == "remove":
            records.remove(match)
            if path == KNOWLEDGE:
                doc["edges"] = [e for e in doc["edges"]
                                if item["node_id"] not in (e["source"],e["target"])]
            continue
        if path == KNOWLEDGE:
            flow.need("type" not in after or after["type"] == match["type"],
                      "Changing scientific node kind is not a content review")
            for key in CONTENT_FIELDS - {"type","status"}:
                if key in after: match[key] = after[key]
            evidence = [x["path"] for x in plan["implementation"]["files"]]
            evidence += [result["result"]["path"] for result in result_records(plan, item["key"])]
            match["evidence"] = sorted(set(match.get("evidence",[]) + evidence))
            match["status"] = "implemented"
        else:
            match["term"] = after["title"]
            match["definition"] = after["description"]
            for key in TERM_FIELDS - {"term", "definition"}:
                if key in after: match[key] = after[key]
    if editing:
        refresh_status(candidates[KNOWLEDGE])
        validate_scientific(candidates[KNOWLEDGE], candidates[GLOSSARY])
        validate_atlas_vocabulary(candidates[VOCAB], candidates[GLOSSARY])
    return {path: candidates[path] for path in files if candidates[path] != originals[path]}


def verify_code(plan: dict) -> None:
    accepted = [item for item in plan["items"]
                if item["view"] == "code" and item["decision"] == "approve"]
    if not accepted:
        return
    from harness.code.build import build
    graph = build()
    by_id = {node["id"]: node for node in graph["nodes"]}
    for item in accepted:
        key = item["node_id"]
        action = item["operation"]
        source = (item.get("before") or item.get("after") or {}).get("path")
        actual = by_id.get(key)
        if action == "remove":
            flow.need(actual is None and flow.code_ast_fingerprint(key, source) is None,
                      "Removed Code node still exists in the current source: " + key)
        else:
            flow.need(actual is not None and actual.get("path") == source,
                      "Accepted Code node is missing or moved unexpectedly: " + key)
            fingerprint = flow.code_ast_fingerprint(key, source)
            flow.need(fingerprint is not None, "Code node lacks actual AST: " + key)
            if action == "change":
                flow.need(item["before_ast_sha256"] not in (None, fingerprint),
                          "Code change did not affect the reviewed function/class/module AST: " + key)


def verify_environment(plan: dict) -> None:
    accepted = [row for row in plan["items"]
                if row["view"] == "environment" and row["decision"] == "approve"]
    if not accepted: return
    for item in accepted:
        profiles = []
        for record in result_records(plan, item["key"]):
            result = record["result"]["path"]
            value = flow.read(flow.ROOT / flow.canonical(result, True))
            if value.get("kind") == "environment_profile" and value.get("synthetic") is False:
                profiles.append(value)
        flow.need(any(value.get("captured_at_utc") for value in profiles),
                  "Approved environment needs a genuine local probe tied to its node")


def apply(plan: dict) -> list[str]:
    # First validate all pending changes in memory. If anything fails, no
    # authoritative graph gets modified; callers keep the plan for re-review.
    verify_code(plan)
    verify_environment(plan)
    updated = validated_updates(plan)
    old_bytes = {path: (flow.ROOT/path).read_bytes() for path in updated}
    committed = []
    try:
        for path,obj in updated.items():
            flow.write(flow.ROOT/path, obj)
            committed.append(path)
    except Exception:
        for path in committed:
            (flow.ROOT/path).write_bytes(old_bytes[path])
        raise
    return list(updated)
