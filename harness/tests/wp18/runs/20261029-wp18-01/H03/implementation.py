"""核对测验用术语引用。只给出修复建议，不写正式 Glossary 或 Knowledge。"""

from __future__ import annotations

import re


class ContextLinkError(ValueError):
    """测验数据包本身无法检查。"""


_EVIDENCE = re.compile(r"^[\w./-]+\:\d+$")


def check_context_links(bundle: dict) -> dict:
    """检查 term_ids、反向 related_node_ids 和可定位的 source_evidence。

    返回诊断和候选建议。同义词不会被合并成新的权威术语，函数也不写文件。
    """
    if not isinstance(bundle, dict) or bundle.get("synthetic") is not True:
        raise ContextLinkError("fixture must be an object marked synthetic")
    nodes = bundle.get("nodes")
    glossary = bundle.get("glossary")
    if not isinstance(nodes, list) or not isinstance(glossary, list):
        raise ContextLinkError("nodes and glossary must be lists")
    diagnostics: list[dict[str, str]] = []
    node_ids, node_terms, node_evidence = _nodes(nodes, diagnostics)
    terms, aliases = _terms(glossary, diagnostics)
    _aliases(aliases, diagnostics)
    _citations(node_ids, node_terms, node_evidence, terms, diagnostics)
    diagnostics.sort(key=lambda item: (item["code"], item.get("node_id", ""), item.get("term_id", ""), item.get("detail", "")))
    return {
        "diagnostics": diagnostics,
        "suggestions": _suggestions(diagnostics),
        "glossary_written": False,
        "implemented_nodes": [],
    }


def _nodes(nodes: list, diagnostics: list[dict[str, str]]) -> tuple[set[str], dict[str, list[str]], dict[str, str]]:
    """收集节点 ID、术语引用和证据。"""
    node_ids: set[str] = set()
    node_terms: dict[str, list[str]] = {}
    node_evidence: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise ContextLinkError("node id must be a nonempty string")
        if node["id"] in node_ids:
            diagnostics.append({"code": "duplicate_node_id", "node_id": node["id"], "detail": node["id"]})
        node_ids.add(node["id"])
        terms = node.get("term_ids", [])
        if not isinstance(terms, list) or not all(isinstance(item, str) and item for item in terms):
            raise ContextLinkError("term_ids must be a list of strings")
        node_terms[node["id"]] = terms
        evidence = node.get("source_evidence", "")
        node_evidence[node["id"]] = evidence if isinstance(evidence, str) else ""
    return node_ids, node_terms, node_evidence


def _terms(glossary: list, diagnostics: list[dict[str, str]]) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """收集术语定义。重复 ID 保留为诊断，不覆盖成一条。"""
    terms: dict[str, dict] = {}
    aliases: list[tuple[str, str]] = []
    for entry in glossary:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ContextLinkError("glossary id must be a nonempty string")
        if entry["id"] in terms:
            diagnostics.append({"code": "duplicate_term_id", "term_id": entry["id"], "detail": entry["id"]})
            continue
        related = entry.get("related_node_ids", [])
        if not isinstance(related, list) or not all(isinstance(item, str) and item for item in related):
            raise ContextLinkError("related_node_ids must be a list of strings")
        evidence = entry.get("source_evidence", "")
        terms[entry["id"]] = {
            "related_node_ids": related,
            "source_evidence": evidence if isinstance(evidence, str) else "",
        }
        for alias in entry.get("aliases", []):
            if not isinstance(alias, str) or not alias:
                raise ContextLinkError("aliases must be nonempty strings")
            aliases.append((alias, entry["id"]))
    return terms, aliases


def _aliases(aliases: list[tuple[str, str]], diagnostics: list[dict[str, str]]) -> None:
    """同义词被两个术语同时占用时只报告，不新建第三个权威术语。"""
    owners: dict[str, str] = {}
    for alias, term_id in aliases:
        previous = owners.get(alias)
        if previous is not None and previous != term_id:
            diagnostics.append({
                "code": "duplicate_alias",
                "term_id": term_id,
                "detail": f"{alias} already used by {previous}",
            })
        owners.setdefault(alias, term_id)


def _citations(
    node_ids: set[str],
    node_terms: dict[str, list[str]],
    node_evidence: dict[str, str],
    terms: dict[str, dict],
    diagnostics: list[dict[str, str]],
) -> None:
    """核对双向引用和可定位证据。"""
    for node_id, evidence in node_evidence.items():
        if _EVIDENCE.fullmatch(evidence) is None:
            diagnostics.append({"code": "missing_source_evidence", "node_id": node_id, "detail": evidence or "empty"})
    for term_id, entry in terms.items():
        if _EVIDENCE.fullmatch(entry["source_evidence"]) is None:
            diagnostics.append({"code": "missing_source_evidence", "term_id": term_id, "detail": entry["source_evidence"] or "empty"})
    cited_by_term = {term_id: set(entry["related_node_ids"]) for term_id, entry in terms.items()}
    for node_id, term_list in node_terms.items():
        for term_id in term_list:
            if term_id not in terms:
                diagnostics.append({"code": "missing_term", "node_id": node_id, "term_id": term_id, "detail": term_id})
                continue
            if node_id not in cited_by_term[term_id]:
                diagnostics.append({
                    "code": "reverse_mismatch",
                    "node_id": node_id,
                    "term_id": term_id,
                    "detail": "node cites term but term does not cite node",
                })
    for term_id, related in cited_by_term.items():
        for node_id in related:
            if node_id not in node_ids:
                diagnostics.append({"code": "missing_node", "node_id": node_id, "term_id": term_id, "detail": node_id})
                continue
            if term_id not in node_terms[node_id]:
                diagnostics.append({
                    "code": "reverse_mismatch",
                    "node_id": node_id,
                    "term_id": term_id,
                    "detail": "term cites node but node does not cite term",
                })


def _suggestions(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    """把诊断转成未执行的修复建议。"""
    suggestions = []
    for item in diagnostics:
        if item["code"] == "duplicate_alias":
            action = "保留已有权威术语，不要把同义词登记成第二个术语"
        elif item["code"] == "duplicate_term_id":
            action = "保留一个术语 ID，不要覆盖或复制定义"
        else:
            action = "只在人工确认证据后修正引用，不要把未验证节点标成 implemented"
        suggestions.append({"code": item["code"], "action": action, "applied": "false"})
    return suggestions
