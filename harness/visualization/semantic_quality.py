"""Validate the five source-backed Atlas trees and report semantic collisions.

Similarity is a review signal, not proof that two scientific functions are equal.
All reported examples retain tree, parent, and node identities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import unicodedata

EDGES = {"contains", "refines", "addresses", "decomposes"}
PUNCTUATION = re.compile(r"[\\s\\W_]+", re.UNICODE)
GENERIC_CODE = "源码依据和调用参数可在节点的展开说明中查看"


def normalized(value: str) -> str:
    return PUNCTUATION.sub("", unicodedata.normalize("NFKC", str(value or "")).casefold())


def similarity(left: str, right: str) -> float:
    a, b = normalized(left), normalized(right)
    if len(a) < 8 or len(b) < 8:
        return 0.0
    x = {a[i:i + 2] for i in range(len(a) - 1)}
    y = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(x & y) / max(1, len(x | y))


def validate_all_trees(graphs: dict) -> dict:
    expected = {"core", "knowledge", "code", "environment", "context"}
    if set(graphs) != expected:
        raise ValueError("Five-tree scope mismatch: " + repr(set(graphs) ^ expected))
    hard = []
    reports = {}
    all_term_ids = {n["id"] for n in graphs["context"]["nodes"]}
    for view, graph in graphs.items():
        by_id = {node["id"]: node for node in graph["nodes"]}
        counts = Counter()
        examples = defaultdict(list)
        children = defaultdict(list)
        if len(by_id) != len(graph["nodes"]):
            hard.append((view, "duplicate_ids"))
        for node in graph["nodes"]:
            id_ = node["id"]
            title, summary = node.get("title", ""), node.get("summary", "")
            short_title, short_summary = normalized(title), normalized(summary)
            if not short_title or not short_summary:
                hard.append((view, id_, "missing title or card summary"))
            if short_title and short_title == short_summary:
                hard.append((view, id_, "title equals card summary"))
            if short_title and short_title == normalized(node.get("description", "")):
                hard.append((view, id_, "title equals description"))
            if normalized(node.get("reader_explanation", "")) == short_summary and short_summary:
                counts["repeated_expanded_text"] += 1
                if len(examples["repeated_expanded_text"]) < 12:
                    examples["repeated_expanded_text"].append(id_)
            if view == "code" and GENERIC_CODE in str(summary):
                counts["template_summary"] += 1
            if node.get("presentation_only"):
                refs = node.get("context_term_ids") or []
                if view != "context" and not refs:
                    hard.append((view, id_, "generated node without Context"))
                if any(ref not in all_term_ids for ref in refs):
                    hard.append((view, id_, "missing authoritative Context term"))
            if ":nav:" in id_ or ":public-nav:" in id_:
                hard.append((view, id_, "fabricated navigation"))
        for edge in graph.get("edges") or []:
            parent, child = edge["source"], edge["target"]
            if parent not in by_id or child not in by_id:
                hard.append((view, parent, child, "unresolved hierarchy endpoint"))
            if edge["type"] in EDGES:
                children[parent].append(child)
        for parent, members in children.items():
            groups = defaultdict(list)
            for id_ in members:
                node = by_id.get(id_)
                if node:
                    groups[(normalized(node.get("title")), normalized(node.get("summary")))].append(id_)
            for pair, ids in groups.items():
                if len(ids) > 1:
                    hard.append((view, parent, "same sibling title and summary", ids))
                    counts["exact_duplicate_sibling_groups"] += 1
                    if len(examples["exact_duplicate_sibling_groups"]) < 60:
                        examples["exact_duplicate_sibling_groups"].append({
                            "parent": parent, "ids": ids,
                        })
                    # One pair of identical text does not by itself prove
                    # equal scientific meaning; it is a consolidation candidate.
            for i, left_id in enumerate(members):
                left = by_id.get(left_id)
                if not left:
                    continue
                for right_id in members[i + 1:]:
                    right = by_id.get(right_id)
                    if not right:
                        continue
                    score = similarity(left.get("summary", ""), right.get("summary", ""))
                    if score >= 0.82 and normalized(left.get("summary")) != normalized(right.get("summary")):
                        counts["similar_sibling_purposes"] += 1
                        if len(examples["similar_sibling_purposes"]) < 12:
                            examples["similar_sibling_purposes"].append({
                                "parent": parent, "ids": [left_id, right_id],
                                "score": round(score, 3),
                            })
        reports[view] = {
            "nodes": len(by_id),
            "counts": dict(counts),
            "examples": dict(examples),
        }
    if hard:
        raise ValueError("Atlas semantic contract violation: " + repr(hard[:30]))
    return reports


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = (root / "harness/visualization/data.js").read_text(encoding="utf-8")
    graphs = json.loads(source.partition("window.OPENRESEARCH_ATLAS = ")[2].strip().removesuffix(";"))
    print("FIVE_TREE_SEMANTICS=" + json.dumps(validate_all_trees(graphs), ensure_ascii=False))
