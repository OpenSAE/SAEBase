from __future__ import annotations

from pathlib import Path

from src.schema import require_keys, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"nodes", "terms", "update"})
    unique([row["id"] for row in payload["nodes"]])
    unique([row["id"] for row in payload["terms"]])
    nodes = {row["id"]: row for row in payload["nodes"]}
    terms = {row["id"]: row for row in payload["terms"]}
    update = payload["update"]
    term_id, node_id = update["term"], update["node"]
    if term_id not in terms or node_id not in nodes:
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
        for lang in ("zh", "en"):
            text = row[lang]
            if (
                not isinstance(text, str)
                or not text.strip()
                or "unavailable" in text.lower()
                or "翻译不可用" in text
            ):
                raise ValueError("translation")
    return {
        "node_ids": sorted(nodes),
        "term_ids": sorted(terms),
        "links": sorted(f"{nid}:{term}" for nid, row in nodes.items() for term in row["terms"]),
        "updated": {
            "term": term_id,
            "node": node_id,
            "zh": terms[term_id]["zh"],
            "en": terms[term_id]["en"],
        },
    }
