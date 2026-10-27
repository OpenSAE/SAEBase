"""Human proposal decisions in the existing Atlas, without changing Core.

Every click is a validated, synchronous write. The review ledger stores pending,
approved and denied decisions. Approved empty Knowledge nodes are published as
not_implemented (RED), never as implemented. Denied proposals remain in the
review ledger (GRAY) and are absent from the Knowledge authority.
"""

from __future__ import annotations

import copy
import hashlib
import json

from harness.core import agent_flow as flow
from harness.knowledge.status import refresh_status
from harness.knowledge.validate import validate_scientific

KNOWLEDGE = "doc/knowledge/context_graph.json"
GLOSSARY = "doc/knowledge/glossary.json"
STATUS = {"approved": "not_implemented", "denied": "denied", "unchecked": "unchecked"}


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_context(plan: dict) -> dict:
    """Only the parent chains of touched Knowledge nodes, not an extra graph."""
    graph = flow.read(flow.ROOT / KNOWLEDGE)
    mapping = {n["id"]: n for n in graph["nodes"]}
    previous = {e["target"]: e for e in graph["edges"]}
    nodes, edges = {}, {}
    for item in plan["items"]:
        if item["view"] != "knowledge":
            continue
        cursor = item.get("parent_id") or previous.get(item["node_id"], {}).get("source")
        visited = set()
        while cursor and cursor in mapping and cursor not in visited:
            visited.add(cursor)
            source = mapping[cursor]
            nodes[cursor] = {
                "id": cursor,
                "kind": source.get("type", "goal"),
                "title": source["title"],
                "description": source["description"],
                "status": source.get("status", "partial"),
            }
            edge = previous.get(cursor)
            if edge:
                edges[(edge["source"], cursor)] = dict(edge)
                cursor = edge["source"]
            else:
                edges[("knowledge:existing", cursor)] = {
                    "source": "knowledge:existing",
                    "target": cursor,
                    "type": "contains",
                }
                break
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def snapshot(plan: dict) -> dict:
    return {"plan": plan, "context": selected_context(plan)}


def _adopt_empty_knowledge(
    graph: dict, glossary: dict, plan: dict, item: dict, decision: str
) -> None:
    if item["view"] != "knowledge" or item["operation"] != "add":
        # Other tree kinds retain review decisions in the JSON ledger; they do
        # not gain unverified scientific/source implementation in this stage.
        return
    files = set(item.get("files") or ())
    flow.need(
        {KNOWLEDGE, GLOSSARY} <= files <= set(plan["allowed_paths"]),
        "Knowledge insertion requires both authorized graph and glossary paths",
    )
    node_id = item["node_id"]
    match = next((n for n in graph["nodes"] if n["id"] == node_id), None)
    if match:
        flow.need(
            match.get("review_origin") == plan["id"]
            and match["status"] == "not_implemented"
            and not match.get("evidence"),
            "Existing node is not a blank proposal owned by this plan",
        )
    if decision in ("denied", "unchecked"):
        if match:
            children = [e for e in graph["edges"] if e["source"] == node_id]
            flow.need(not children, "Cannot deny a placeholder with dependent nodes")
            graph["nodes"].remove(match)
            graph["edges"][:] = [
                e for e in graph["edges"] if node_id not in (e["source"], e["target"])
            ]
            for term in glossary["terms"]:
                if node_id in term["related_node_ids"]:
                    term["related_node_ids"].remove(node_id)
        return
    if match:
        return  # Reopening an already approved proposal is idempotent.
    after = item.get("after") or {}
    kind, parent_id = after.get("type"), item.get("parent_id")
    flow.need(
        kind in ("goal", "question", "solve") and isinstance(parent_id, str),
        "Knowledge proposal must declare a typed node and parent",
    )
    parents = [n for n in graph["nodes"] if n["id"] == parent_id]
    flow.need(len(parents) == 1, "Knowledge parent does not exist")
    relation = {
        ("question", "question"): "refines",
        ("question", "solve"): "addresses",
        ("solve", "goal"): "decomposes",
        ("goal", "goal"): "decomposes",
    }.get((parents[0]["type"], kind))
    flow.need(relation is not None, "Invalid Knowledge tree parent/child types")
    terms = after.get("term_ids")
    flow.need(
        isinstance(terms, list) and bool(terms) and len(terms) == len(set(terms)),
        "Knowledge placeholder must reference defined scientific terms",
    )
    vocab = {term["id"]: term for term in glossary["terms"]}
    flow.need(set(terms) <= set(vocab), "Knowledge term reference not in glossary")
    flow.need(
        after.get("evidence", []) == [], "Empty proposal cannot invent implementation evidence"
    )
    graph["nodes"].append(
        {
            "id": node_id,
            "type": kind,
            "title": after["title"],
            "description": after["description"],
            "status": "not_implemented",
            "evidence": [],
            "term_ids": list(terms),
            "environment_refs": [],
            "next": "待实现",
            "review_origin": plan["id"],
        }
    )
    graph["edges"].append({"source": parent_id, "target": node_id, "type": relation})
    for key in terms:
        vocab[key]["related_node_ids"].append(node_id)


def decide(plan: dict, key: str, decision: str, *, actor: str = "Local human") -> dict:
    """Persist decision, authority graph and glossary before returning success."""
    flow.validate_seal(plan)
    flow.need(
        plan["phase"] in ("preflight", "proposal_reviewing", "proposal_reviewed"),
        "Only unimplemented proposal review can use these controls",
    )
    flow.need(
        decision in STATUS and isinstance(actor, str) and actor.strip(),
        "Select human approval or denial",
    )
    match = [item for item in plan["items"] if item["key"] == key]
    flow.need(len(match) == 1, "Unknown review node")
    item = match[0]
    # Explicit decisions may be revisited; only a safely owned blank node can
    # be removed again. No code, ML outcome or Core status is ever greened.
    expected = plan.get("review_source_sha256") or plan["authority_digest"].get("knowledge")
    expected_glossary = plan.get("review_glossary_sha256") or plan["file_sha256_before"].get(
        GLOSSARY
    )
    graph_path, glossary_path = flow.ROOT / KNOWLEDGE, flow.ROOT / GLOSSARY
    old_graph, old_glossary = graph_path.read_bytes(), glossary_path.read_bytes()
    if item["view"] == "knowledge" and item["operation"] == "add":
        flow.need(
            fingerprint(old_graph) == expected and fingerprint(old_glossary) == expected_glossary,
            "Knowledge/Glossary changed outside this review; refresh audit",
        )
    graph, glossary = json.loads(old_graph), json.loads(old_glossary)
    _adopt_empty_knowledge(graph, glossary, plan, item, decision)
    if item["view"] == "knowledge" and item["operation"] == "add":
        refresh_status(graph)
        validate_scientific(graph, glossary)
    original_plan = copy.deepcopy(plan)
    item["proposal_review"] = (
        None
        if decision == "unchecked"
        else {"decision": decision, "actor": actor.strip(), "at": flow.now()}
    )
    completed = sum(bool(row.get("proposal_review")) for row in plan["items"])
    plan["phase"] = (
        "proposal_reviewed"
        if completed == len(plan["items"])
        else "preflight"
        if completed == 0
        else "proposal_reviewing"
    )
    next_graph = json.dumps(graph, ensure_ascii=False, indent=2) + "\n"
    next_glossary = json.dumps(glossary, ensure_ascii=False, indent=2) + "\n"
    # Persist the validated authority first; then record exactly which bytes
    # the next review may modify. Roll back normal write errors.
    try:
        if next_graph.encode() != old_graph:
            flow.write(graph_path, graph)
        if next_glossary.encode() != old_glossary:
            flow.write(glossary_path, glossary)
        if item["view"] == "knowledge" and item["operation"] == "add":
            plan["review_source_sha256"] = flow.checksum(KNOWLEDGE)
            plan["review_glossary_sha256"] = flow.checksum(GLOSSARY)
        flow.save(plan, "proposal_review", key=key, decision=decision, actor=actor.strip())
    except Exception:
        graph_path.write_bytes(old_graph)
        glossary_path.write_bytes(old_glossary)
        plan.clear()
        plan.update(original_plan)
        raise
    return snapshot(plan)
