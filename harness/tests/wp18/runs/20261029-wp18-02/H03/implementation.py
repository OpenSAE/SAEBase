"""核对测验包里的术语引用。路径字符串本身不算已经核实的来源。"""

from __future__ import annotations

from pathlib import Path


class LinkBundleError(ValueError):
    """数据包的形状无法检查。"""


def audit_term_links(bundle: dict, evidence_root: str | Path) -> dict:
    """检查 term_ids、反向关联，并把来源分成已解析、未解析或拒绝。

    只返回建议。不会写 Glossary，也不会把同义词升级成新术语。
    """
    if not isinstance(bundle, dict) or bundle.get("synthetic") is not True:
        raise LinkBundleError("bundle must be an object marked synthetic")
    root = Path(evidence_root).resolve()
    if not root.is_dir():
        raise LinkBundleError("evidence root is not a directory")
    nodes = bundle.get("nodes")
    glossary = bundle.get("glossary")
    if not isinstance(nodes, list) or not isinstance(glossary, list):
        raise LinkBundleError("nodes and glossary must be lists")
    diagnostics: list[dict[str, str]] = []
    node_ids, node_terms = _read_nodes(nodes, diagnostics)
    terms, aliases = _read_terms(glossary, diagnostics)
    _alias_collisions(aliases, diagnostics)
    provenance = []
    for owner, reference in _evidence_pairs(nodes, glossary):
        provenance.append(_resolve_reference(root, owner, reference, diagnostics))
    _link_diagnostics(node_ids, node_terms, terms, diagnostics)
    diagnostics.sort(key=lambda item: (item["code"], item.get("owner", ""), item.get("detail", "")))
    return {
        "diagnostics": diagnostics,
        "provenance": provenance,
        "suggestions": _advice(diagnostics),
        "glossary_written": False,
        "implemented_nodes": [],
    }


def _read_nodes(nodes: list, diagnostics: list[dict[str, str]]) -> tuple[set[str], dict[str, list[str]]]:
    """收集节点 ID 和它引用的术语。重复 ID 只记诊断，不覆盖。"""
    node_ids: set[str] = set()
    node_terms: dict[str, list[str]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise LinkBundleError("node id must be a nonempty string")
        if node["id"] in node_ids:
            diagnostics.append({"code": "duplicate_node_id", "owner": node["id"], "detail": node["id"]})
        node_ids.add(node["id"])
        terms = node.get("term_ids", [])
        if not isinstance(terms, list) or not all(isinstance(item, str) and item for item in terms):
            raise LinkBundleError("term_ids must be a list of strings")
        node_terms[node["id"]] = list(terms)
    return node_ids, node_terms


def _read_terms(glossary: list, diagnostics: list[dict[str, str]]) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """收集术语。后出现的同一 ID 不覆盖先出现的定义。"""
    terms: dict[str, dict] = {}
    aliases: list[tuple[str, str]] = []
    for entry in glossary:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise LinkBundleError("glossary id must be a nonempty string")
        if entry["id"] in terms:
            diagnostics.append({"code": "duplicate_term_id", "owner": entry["id"], "detail": entry["id"]})
            continue
        related = entry.get("related_node_ids", [])
        if not isinstance(related, list) or not all(isinstance(item, str) and item for item in related):
            raise LinkBundleError("related_node_ids must be a list of strings")
        terms[entry["id"]] = {"related_node_ids": list(related)}
        for alias in entry.get("aliases", []):
            if not isinstance(alias, str) or not alias:
                raise LinkBundleError("aliases must be nonempty strings")
            aliases.append((alias, entry["id"]))
    return terms, aliases


def _alias_collisions(aliases: list[tuple[str, str]], diagnostics: list[dict[str, str]]) -> None:
    """两个术语共用别名时只报告，不创建第三个权威术语。"""
    owners: dict[str, str] = {}
    for alias, term_id in aliases:
        previous = owners.get(alias)
        if previous is not None and previous != term_id:
            diagnostics.append({
                "code": "duplicate_alias",
                "owner": term_id,
                "detail": f"{alias} already used by {previous}",
            })
        owners.setdefault(alias, term_id)


def _evidence_pairs(nodes: list, glossary: list) -> list[tuple[str, object]]:
    """按输入顺序取出每条 source_evidence，供来源解析使用。"""
    pairs = []
    for node in nodes:
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            pairs.append((node["id"], node.get("source_evidence", "")))
    for entry in glossary:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            pairs.append((entry["id"], entry.get("source_evidence", "")))
    return pairs


def _resolve_reference(root: Path, owner: str, reference: object, diagnostics: list[dict[str, str]]) -> dict[str, str]:
    """区分句法拒绝、文件不存在和真正读到的行。"""
    if not isinstance(reference, str) or reference.count(":") != 1:
        diagnostics.append({"code": "rejected_reference", "owner": owner, "detail": "unparseable"})
        return {"owner": owner, "reference": str(reference), "status": "rejected"}
    relative, line_text = reference.split(":", 1)
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        diagnostics.append({"code": "rejected_reference", "owner": owner, "detail": relative or "empty"})
        return {"owner": owner, "reference": reference, "status": "rejected"}
    if not line_text.isdigit() or int(line_text) < 1:
        diagnostics.append({"code": "bad_line", "owner": owner, "detail": line_text or "empty"})
        return {"owner": owner, "reference": reference, "status": "rejected"}
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink() and not current.resolve().is_relative_to(root):
            diagnostics.append({"code": "rejected_reference", "owner": owner, "detail": relative})
            return {"owner": owner, "reference": reference, "status": "rejected"}
    if not current.is_file():
        diagnostics.append({"code": "provenance_unresolved", "owner": owner, "detail": relative})
        return {"owner": owner, "reference": reference, "status": "unresolved"}
    line_count = len(current.read_text(encoding="utf-8").splitlines())
    if int(line_text) > line_count:
        diagnostics.append({"code": "bad_line", "owner": owner, "detail": line_text})
        return {"owner": owner, "reference": reference, "status": "rejected"}
    return {"owner": owner, "reference": reference, "status": "resolved"}


def _link_diagnostics(
    node_ids: set[str],
    node_terms: dict[str, list[str]],
    terms: dict[str, dict],
    diagnostics: list[dict[str, str]],
) -> None:
    """核对双向引用。缺失的一端不会被自动补上。"""
    for node_id, term_list in node_terms.items():
        for term_id in term_list:
            if term_id not in terms:
                diagnostics.append({"code": "missing_term", "owner": node_id, "detail": term_id})
                continue
            if node_id not in terms[term_id]["related_node_ids"]:
                diagnostics.append({
                    "code": "reverse_mismatch",
                    "owner": node_id,
                    "detail": f"{node_id} cites {term_id} without a reverse link",
                })
    for term_id, entry in terms.items():
        for node_id in entry["related_node_ids"]:
            if node_id not in node_ids:
                diagnostics.append({"code": "missing_node", "owner": term_id, "detail": node_id})
                continue
            if term_id not in node_terms[node_id]:
                diagnostics.append({
                    "code": "reverse_mismatch",
                    "owner": term_id,
                    "detail": f"{term_id} cites {node_id} without a reverse link",
                })


def _advice(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    """把诊断变成尚未执行的建议。"""
    advice = []
    for item in diagnostics:
        if item["code"] == "duplicate_alias":
            action = "保留已有术语，不要把同义词登记成新的权威 ID"
        elif item["code"] == "provenance_unresolved":
            action = "来源尚未解析，保持 unresolved，不要把它写成已核实"
        else:
            action = "只在人工确认后修正引用，不要把未验证节点标成 implemented"
        advice.append({"code": item["code"], "action": action, "applied": "false"})
    return advice
