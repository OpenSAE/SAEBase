"""Audit display responsibilities against actual Code Tree parentage.

This prints every distinct problem category with source IDs. It deliberately
does not rewrite titles or infer behavior from Python symbol spelling.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def normalize(text: str) -> str:
    return re.sub(r"[\\W_]+", "", str(text or ""), flags=re.UNICODE).casefold()


def audit(graphs: dict) -> dict:
    graph = graphs["code"]
    by_id = {n["id"]: n for n in graph["nodes"]}
    children = defaultdict(list)
    for edge in graph["edges"]:
        if edge["type"] == "contains":
            children[edge["source"]].append(edge["target"])
    summary_counts = Counter(normalize(n.get("summary", "")) for n in by_id.values())
    title_counts = Counter(normalize(n.get("title", "")) for n in by_id.values())
    issues = defaultdict(list)
    for node in by_id.values():
        title = normalize(node.get("title"))
        summary = normalize(node.get("summary"))
        details = normalize(node.get("reader_explanation"))
        if summary and details and (summary == details or summary in details):
            issues["duplicated_explanation"].append(node["id"])
        if summary.startswith("负责") and "源码依据和调用参数" in node.get("summary", ""):
            issues["template_summary"].append(node["id"])
        if title and summary and (title == summary or summary == "负责" + title):
            issues["tautological_purpose"].append(node["id"])
    for parent, ids in children.items():
        for field in ("title", "summary"):
            texts = defaultdict(list)
            for key in ids:
                norm = normalize(by_id[key].get(field))
                if norm:
                    texts[norm].append(key)
            for keys in texts.values():
                if len(keys) > 1:
                    issues["repeated_sibling_" + field].append(
                        {"parent": parent, "members": keys}
                    )
        parent_title = normalize(by_id.get(parent, {}).get("title"))
        parent_summary = normalize(by_id.get(parent, {}).get("summary"))
        for key in ids:
            child = by_id[key]
            title = normalize(child.get("title"))
            summary = normalize(child.get("summary"))
            if title and parent_title == title:
                issues["duplicate_parent_title"].append((parent, key))
            if summary and summary == parent_summary:
                issues["duplicate_parent_purpose"].append((parent, key))
    return {
        "code_nodes": len(by_id),
        "counts": {k: len(v) for k, v in issues.items()},
        "top_repeated_summaries": [
            {"count": count, "example": next(
                n["summary"] for n in by_id.values()
                if normalize(n.get("summary")) == value
            )}
            for value, count in summary_counts.most_common(15) if count > 2
        ],
        "sample": {k: v[:12] for k, v in issues.items()},
    }


def main() -> None:
    text = (ROOT / "harness/visualization/data.js").read_text(encoding="utf-8")
    prefix = "window.OPENRESEARCH_ATLAS = "
    if prefix not in text:
        raise ValueError("Missing source-generated Atlas")
    graphs = json.loads(text.partition(prefix)[2].strip().removesuffix(";"))
    print("ATLAS_SEMANTIC_AUDIT=" + json.dumps(audit(graphs), ensure_ascii=False))


if __name__ == "__main__":
    main()
