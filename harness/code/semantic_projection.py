"""Curate the public Code Tree without losing any AST interfaces.

Source entities are not automatically independent research capabilities.
Unreviewed function/class/module cards with generated generic explanations are
kept as expandable source records under the closest genuinely described
research node. No source code, signature, ID, or original AST snapshot is
deleted. The source-backed AST remains in doc/code/code_tree.json.
"""

from __future__ import annotations

from collections import defaultdict
import re

from harness.code.audit_semantics import normalize, purpose
from harness.code.presentation import SOURCE_REVIEWS

GENERIC_PHRASES = (
    "源码依据和调用参数",
    "按相近的科研职责组织已有接口",
)
HIERARCHY = "contains"


def _source_record(node: dict) -> dict:
    """Preserve original IDs and code; omit unverified callable paraphrases."""
    import copy

    attributes = copy.deepcopy(node.get("attributes") or [])
    for block in attributes:
        interfaces = (
            [block["interface"]]
            if block.get("type") == "interface"
            else block.get("methods", [])
            if block.get("type") == "class_methods"
            else []
        )
        for info in interfaces:
            # Generated Chinese fallback prose has no independently verified
            # English translation. Exact signatures, returns and code remain.
            if any(0x3400 <= ord(c) <= 0x9FFF for c in info.get("purpose", "")):
                info.pop("purpose", None)
    return {
        "id": node["id"],
        "kind": node.get("kind", ""),
        "source_symbol": node.get("source_symbol", ""),
        "path": node.get("path") or node.get("source_path") or "",
        "evidence": node.get("evidence") or [],
        "line": node.get("line"),
        "signature": node.get("signature", ""),
        "context_term_ids": node.get("context_term_ids") or [],
        "attributes": attributes,
    }


def _independently_described(node: dict) -> bool:
    """Require an actual, distinguishable role, not symbol-based decoration."""
    if node.get("presentation_only"):
        return True
    if node.get("id") in SOURCE_REVIEWS:
        return True
    title = str(node.get("title") or "")
    summary = str(node.get("summary") or "")
    if not title or not summary or any(word in summary for word in GENERIC_PHRASES):
        return False
    return normalize(title) != normalize(summary) and purpose(title) != purpose(summary)


def compact_public_code_tree(graph: dict) -> dict:
    """Fold source API details into reviewed capabilities while retaining IDs.

    Reviewed interfaces remain navigable cards. Other AST nodes are attached
    as inspectable source records under their nearest visible conceptual owner.
    Exact/near duplicate sibling cards are consolidated, never renamed with
    a source filename just to satisfy a uniqueness assertion.
    """
    nodes = {node["id"]: node for node in graph["nodes"]}
    root = next(n["id"] for n in graph["nodes"] if n.get("kind") == "root")
    parents = {}
    children = defaultdict(list)
    for edge in graph["edges"]:
        if edge.get("type") == HIERARCHY:
            child, parent = edge["target"], edge["source"]
            if child in parents and parents[child] != parent:
                raise ValueError("Ambiguous source graph parent: " + child)
            parents[child] = parent
            children[parent].append(child)

    keep = {
        node_id
        for node_id, node in nodes.items()
        if node_id == root or _independently_described(node)
    }
    # Only semantic collision inside one parent qualifies for automatic
    # folding; a reviewed node's distinct verified role is never erased.
    for parent, ids in children.items():
        duplicates = defaultdict(list)
        for child in ids:
            if child in keep:
                node = nodes[child]
                fingerprint = (normalize(node.get("title")), normalize(node.get("summary")))
                if all(fingerprint):
                    duplicates[fingerprint].append(child)
        for members in duplicates.values():
            if len(members) < 2:
                continue
            reviewed = [x for x in members if x in SOURCE_REVIEWS]
            if len(reviewed) > 1:
                raise ValueError("Reviewed sibling responsibilities collide: " + repr(reviewed))
            # If a curated concept already represents the same role, its
            # actual source modules belong in the concept's interface list.
            # Otherwise retain exactly one source-backed responsibility card.
            canonical = (
                reviewed[0]
                if reviewed
                else None
                if parent in keep and nodes[parent].get("concept_group")
                else members[0]
            )
            for member in members:
                if member != canonical:
                    keep.discard(member)

    # Alternative parameter scales of the same model are concrete source
    # profiles, not separate scientific operations. The already reviewed
    # model-family concept owns their individual configuration interfaces.
    for parent, ids in children.items():
        variants = defaultdict(list)
        for child in ids:
            if child.startswith("code:instance:models:"):
                slug = child.rsplit(":", 1)[-1]
                family = re.sub(r"_[0-9]+(?:p[0-9]+)?b$", "", slug)
                if family != slug:
                    variants[family].append(child)
        if parent in keep:
            for members in variants.values():
                if len(members) > 1:
                    for member in members:
                        keep.discard(member)

    def visible_parent(node_id: str) -> str:
        seen = set()
        current = parents.get(node_id)
        while current not in keep:
            if current is None:
                # Excluded __init__.py forwarding modules can leave AST class
                # records without a visible source parent. Keep these
                # records under the real top-level source area.
                path = nodes[node_id].get("path") or ""
                area = path.split("/", 1)[0]
                owner = "code:group:" + area
                return owner if owner in keep else root
            if current in seen:
                raise ValueError("Cycle in compacted source: " + node_id)
            seen.add(current)
            current = parents.get(current)
        return current

    # Folding a navigation instance can bring its former children together
    # under one concept. Re-audit the *final* parents until no newly adjacent
    # source interfaces repeat. This prevents post-compaction collisions.
    changed = True
    while changed:
        changed = False
        visible_children = defaultdict(list)
        for node_id in keep:
            if node_id != root:
                visible_children[visible_parent(node_id)].append(node_id)
        for parent, ids in visible_children.items():
            collisions = defaultdict(list)
            for node_id in ids:
                node = nodes[node_id]
                key = (normalize(node.get("title")), normalize(node.get("summary")))
                if all(key):
                    collisions[key].append(node_id)
            for group in collisions.values():
                if len(group) < 2:
                    continue
                reviewed = [item for item in group if item in SOURCE_REVIEWS]
                if len(reviewed) > 1:
                    raise ValueError(
                        "Duplicate independently reviewed responsibilities: " + repr(reviewed)
                    )
                retained = (
                    reviewed[0]
                    if reviewed
                    else (None if nodes[parent].get("concept_group") else sorted(group)[0])
                )
                for item in group:
                    if item != retained:
                        keep.discard(item)
                        changed = True

    originals = set(nodes)
    source_records = defaultdict(list)
    mapping = {}
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id in keep:
            continue
        target = visible_parent(node_id)
        source_records[target].append(_source_record(node))
        mapping[node_id] = target

    # Do not repeat a summary as the expanded description. The original AST
    # source and actual interfaces live in attributes, not invented prose.
    for node_id in keep:
        node = nodes[node_id]
        if normalize(node.get("summary")) == normalize(node.get("description")):
            node["description"] = ""
        if source_records[node_id]:
            node.setdefault("attributes", []).append(
                {
                    "type": "source_interfaces",
                    "label": "源码接口与实现依据 · " + str(len(source_records[node_id])) + " 项",
                    "members": source_records[node_id],
                }
            )
    source_edges = []
    for node_id in keep:
        if node_id != root:
            source_edges.append(
                {
                    "type": HIERARCHY,
                    "source": visible_parent(node_id),
                    "target": node_id,
                }
            )
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] in keep]
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if edge.get("type") != HIERARCHY
        and edge.get("source") in keep
        and edge.get("target") in keep
    ] + source_edges
    graph["source_api_index"] = mapping
    graph["source_api_total"] = len(originals)
    if len(mapping) + len(graph["nodes"]) != len(originals):
        raise ValueError("Source API coverage changed during semantic compaction")
    return graph
