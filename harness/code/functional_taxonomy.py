"""Build a source-backed conceptual Code hierarchy without inventing Python APIs.

The concept catalog is reviewed data, keyed to authoritative Context terms.
Category membership must be exhaustive and unambiguous for every real module.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "harness/code/atlas_concepts.json"
PARENT = "code:group:saebase"


# Pure compatibility re-exports preserve historical Python imports but do not
# implement independent scientific operations. The source remains in GitHub;
# Atlas represents the canonical implementation instead.
REEXPORT_DOCSTRINGS = frozenset(
    {
        "Legacy import forwarding to the canonical Dataset implementation.",
        "Compatibility import for Dataset plugin registration.",
        "Compatibility import of the unified Task registry.",
        "Compatibility re-export of the unified model plugin module.",
    }
)


def is_reexport_only(node: dict) -> bool:
    return (
        node.get("kind") == "module"
        and str(node.get("description") or "").strip() in REEXPORT_DOCSTRINGS
    )


def _source_area(path: str) -> str:
    pieces = Path(path).parts
    if len(pieces) < 2 or pieces[0] != "saebase":
        raise ValueError("Concept taxonomy received foreign module " + path)
    return pieces[1] if len(pieces) > 2 else "(root)"


def _concept_node(spec: dict, node_id: str, modules: list[dict]) -> dict:
    # A concrete Dataset/Model instance is a navigation parent backed by
    # actual Python files, so its evidence supplies source paths when it has
    # no direct module path of its own.
    paths = list(
        dict.fromkeys(
            str(p).split(":", 1)[0]
            for n in modules
            for p in ([n["path"]] if n.get("path") else n.get("evidence") or [])
            if p
        )
    )
    if not paths:
        raise ValueError("Concept without source implementations: " + node_id)
    if not spec.get("context_terms"):
        raise ValueError("Concept lacks Context: " + node_id)
    if not all(
        spec.get(k)
        for k in ("title", "summary", "description", "en_title", "en_summary", "en_description")
    ):
        raise ValueError("Incomplete conceptual responsibility: " + node_id)
    return {
        "id": node_id,
        "kind": "group",
        "title": spec["title"],
        "summary": spec["summary"],
        "description": spec["description"],
        "english_title": spec["en_title"],
        "english_summary": spec["en_summary"],
        "english_description": spec["en_description"],
        "concept_group": True,
        "status": "implemented",
        "context_term_ids": spec["context_terms"],
        "evidence": paths,
        "source_symbol": "",
        "presentation_only": True,
    }


def organize_saebase_concepts(output: dict) -> dict:
    """Replace saebase's flat module children with meaningful functional nodes."""
    config = json.loads(CATALOG.read_text(encoding="utf-8"))
    if config.get("schema") != "saebase.atlas.concepts.v1":
        raise ValueError("Unexpected SAEBase concept catalog")
    max_children = config["max_children"]
    preferred = config["preferred_children"]
    if not (1 <= preferred < max_children <= 20):
        raise ValueError("Invalid concept hierarchy fanout policy")
    concepts = config["concepts"]
    groups = {concept["name"]: concept for concept in concepts}
    if len(groups) != len(concepts) or not (10 <= len(groups) <= max_children):
        raise ValueError("Require 10-20 unique top-level Code concepts")
    by_id = {n["id"]: n for n in output["nodes"]}
    direct = [e for e in output["edges"] if e["type"] == "contains" and e["source"] == PARENT]
    original_ids = [e["target"] for e in direct]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Duplicate direct source children")
    original_modules = [by_id[x] for x in original_ids]
    if any(n.get("kind") != "module" or not n.get("path") for n in original_modules):
        raise ValueError("Expected only real Python modules under " + PARENT)
    # All source modules preserve their IDs and descendants. Only contains-edges
    # at this one hierarchy level are redirected.
    assignments = defaultdict(list)
    for node in original_modules:
        matches = [c["name"] for c in concepts if _source_area(node["path"]) in c["source_dirs"]]
        if len(matches) != 1:
            raise ValueError(
                "Unclassified or multiply classified module: "
                + node["path"]
                + " -> "
                + repr(matches)
            )
        assignments[matches[0]].append(node)
    if set(assignments) != set(groups):
        raise ValueError(
            "Catalog concepts have no source implementations: "
            + repr(set(groups) - set(assignments))
        )
    output["edges"] = [e for e in output["edges"] if e not in direct]
    created = []
    for category in concepts:
        name = category["name"]
        members = assignments[name]
        category_id = "code:concept:" + name
        if category_id in by_id:
            raise ValueError("Duplicate functional concept ID: " + category_id)
        parent_node = _concept_node(category, category_id, members)
        created.append(parent_node)
        by_id[category_id] = parent_node
        output["edges"].append({"source": PARENT, "target": category_id, "type": "contains"})
        subgroups = category.get("children") or []
        if subgroups:
            subnames = [x["name"] for x in subgroups]
            if len(set(subnames)) != len(subnames) or len(subgroups) > max_children:
                raise ValueError("Invalid subdomain names under " + category_id)
            partition = defaultdict(list)
            for node in members:
                short = node["path"].removeprefix("saebase/")
                # Tooling is organized by executable verification intent; the
                # tools/ package label is not part of a tool's functional name.
                match_path = short.removeprefix("tools/") if name == "verification" else short
                matches = [
                    x["name"]
                    for x in subgroups
                    if any(re.fullmatch(p, match_path) for p in x["patterns"])
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Non-unique source responsibility: " + node["path"] + " -> " + repr(matches)
                    )
                partition[matches[0]].append(node)
            if len(partition) != len(subgroups):
                raise ValueError(
                    "Empty subdomain in "
                    + category_id
                    + ": "
                    + repr(set(subnames) - set(partition))
                )
            for sub in subgroups:
                group_id = category_id + ":" + sub["name"]
                items = partition[sub["name"]]
                if len(items) > max_children:
                    raise ValueError(
                        "Functional subdomain exceeds twenty real modules: "
                        + group_id
                        + " -> "
                        + str(len(items))
                    )
                sub_node = _concept_node(sub, group_id, items)
                created.append(sub_node)
                by_id[group_id] = sub_node
                output["edges"].append(
                    {"source": category_id, "target": group_id, "type": "contains"}
                )
                for item in items:
                    output["edges"].append(
                        {"source": group_id, "target": item["id"], "type": "contains"}
                    )
        else:
            if len(members) > max_children:
                raise ValueError("Unpartitioned functional domain: " + category_id)
            for item in members:
                output["edges"].append(
                    {"source": category_id, "target": item["id"], "type": "contains"}
                )
    output["nodes"].extend(created)
    new_ids = {n["id"] for n in created}
    assert not new_ids.intersection(original_ids)
    # Compare the exact leaf frontier, not total node counts; no real source
    # entity may be dropped, renamed, or duplicated during conceptualization.
    descend = defaultdict(list)
    for edge in output["edges"]:
        if edge["type"] == "contains":
            descend[edge["source"]].append(edge["target"])
    observed = []
    stack = [PARENT]
    while stack:
        current = stack.pop()
        if current in original_ids:
            observed.append(current)
            continue
        stack.extend(reversed(descend[current]))
    if sorted(observed) != sorted(original_ids):
        raise ValueError("Concept hierarchy lost or duplicated a source module")
    for parent in [PARENT, *new_ids]:
        if len(descend[parent]) > max_children:
            raise ValueError("Concept fanout exceeds 20: " + parent)
    output["concept_hierarchy_schema"] = config["schema"]
    return {
        "concepts": len(concepts),
        "semantic_groups": len(created),
        "source_modules": len(original_ids),
    }


# Extra depth is created only for these six reviewed source responsibilities.
# Research Lab test functions are represented by their test module as evidence;
# they are not public capabilities.
DETAIL_SCOPES = {
    "code:module:saebase.registry": "registry",
    "code:module:saebase.research_lab.backend.api": "lab_api",
    "code:module:tasks.benchmark.stage_d": "benchmark_acceptance",
    "code:concrete:models": "model_families",
    "code:instance:tasks:benchmark": "benchmark_task",
}


def organize_oversized_concepts(output: dict) -> None:
    """Break reviewed, overfull branches into actual distinct responsibilities."""
    path = ROOT / "harness/code/oversized_concepts.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema") != "saebase.atlas.oversized.v1":
        raise ValueError("Unexpected detailed concept catalog")
    parents = catalog["parents"]
    if set(x["id"] for x in parents) != set(DETAIL_SCOPES):
        raise ValueError("Detailed concept source ownership changed without review")
    by_id = {n["id"]: n for n in output["nodes"]}
    for record in parents:
        parent = record["id"]
        if parent not in by_id:
            raise ValueError("Missing source parent for concept refinement: " + parent)
        edges = [e for e in output["edges"] if e["type"] == "contains" and e["source"] == parent]
        members = [by_id[e["target"]] for e in edges]
        if len(members) <= catalog["max_children"]:
            raise ValueError("Expected an overfull source responsibility at " + parent)
        groups = record["groups"]
        grouped = defaultdict(list)
        for node in members:
            if record["match_on"] == "symbol":
                lookup = node.get("source_symbol") or node["id"].rsplit(":", 1)[-1]
            elif record["match_on"] == "path_or_id":
                lookup = node.get("path") or node["id"]
            else:
                raise ValueError("Unknown exact source selector")
            hits = [
                sub["name"]
                for sub in groups
                if any(re.fullmatch(regex, lookup) for regex in sub["patterns"])
            ]
            if len(hits) != 1:
                raise ValueError(
                    "Ambiguous detailed source responsibility: " + node["id"] + " -> " + repr(hits)
                )
            grouped[hits[0]].append(node)
        if set(grouped) != set(g["name"] for g in groups):
            raise ValueError("Empty detailed source responsibility at " + parent)
        output["edges"] = [e for e in output["edges"] if e not in edges]
        for sub in groups:
            items = grouped[sub["name"]]
            if len(items) > catalog["max_children"]:
                raise ValueError("Oversized reviewed detail: " + parent + "/" + sub["name"])
            group_id = "code:concept:" + DETAIL_SCOPES[parent] + ":" + sub["name"]
            if group_id in by_id:
                raise ValueError("Repeated semantic concept " + group_id)
            new_node = _concept_node(sub, group_id, items)
            output["nodes"].append(new_node)
            by_id[group_id] = new_node
            output["edges"].append({"source": parent, "target": group_id, "type": "contains"})
            for item in items:
                output["edges"].append(
                    {"source": group_id, "target": item["id"], "type": "contains"}
                )
