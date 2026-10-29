"""审计测验用中英覆盖层。只报告问题，不修改正式翻译表。"""

from __future__ import annotations

import re


class LocaleAuditError(ValueError):
    """覆盖层记录无法检查。"""


_PLACEHOLDER = "english translation unavailable"
_TEMPLATES = {
    "this function implements the requested behavior.",
    "this module handles the feature.",
}
_IDENTIFIERS = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b")


def audit_overlay(records: list[dict]) -> dict:
    """按稳定 ID 检查中英字段、占位文案、乱码、项目 ID 和被拆开的符号。

    detected 是发现的问题。fixed 始终为空，因为本函数不写回翻译。
    """
    if not isinstance(records, list):
        raise LocaleAuditError("records must be a list")
    detected: list[dict[str, str]] = []
    preserved: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("synthetic") is not True:
            raise LocaleAuditError("each record must be marked synthetic")
        node_id = record.get("id")
        signature = record.get("signature")
        path = record.get("path")
        if not all(isinstance(item, str) and item for item in (node_id, signature, path)):
            raise LocaleAuditError("id, signature, and path must be nonempty strings")
        if node_id in seen:
            raise LocaleAuditError(f"duplicate id: {node_id}")
        seen.add(node_id)
        preserved.append({"id": node_id, "signature": signature, "path": path})
        _record(record, detected)
    detected.sort(key=lambda item: (item["id"], item["code"], item["field"]))
    return {
        "detected": detected,
        "fixed": [],
        "preserved": preserved,
        "note": "detected lists problems only; nothing in the authority translation tables was edited",
    }


def _record(record: dict, detected: list[dict[str, str]]) -> None:
    """检查一条覆盖层记录。"""
    node_id = record["id"]
    if record.get("project") != "SAEBase":
        detected.append({"id": node_id, "code": "wrong_project", "field": "project", "detail": str(record.get("project"))})
    localized = record.get("localized_signature")
    if isinstance(localized, str) and localized != record["signature"]:
        detected.append({"id": node_id, "code": "symbol_translated", "field": "localized_signature", "detail": localized})
    for name in ("title", "summary", "description"):
        _field(record, name, detected)
    title = _text(record.get("title"), "en")
    summary = _text(record.get("summary"), "en")
    if title and summary and _norm(title) == _norm(summary):
        detected.append({"id": node_id, "code": "title_repeats_summary", "field": "summary", "detail": summary})
    _symbols(record, detected)


def _field(record: dict, name: str, detected: list[dict[str, str]]) -> None:
    """英文标题、摘要、描述缺项、占位和乱码都记为问题。"""
    value = record.get(name)
    text = _text(value, "en")
    if text == "":
        detected.append({"id": record["id"], "code": "missing_english", "field": name, "detail": "empty"})
        return
    lowered = text.lower()
    if _PLACEHOLDER in lowered or _norm(text) in { _norm(item) for item in _TEMPLATES }:
        detected.append({"id": record["id"], "code": "translation_placeholder", "field": name, "detail": text})
    if "\ufffd" in text or "\ufffd" in _text(value, "zh"):
        detected.append({"id": record["id"], "code": "undecodable_text", "field": name, "detail": text})


def _symbols(record: dict, detected: list[dict[str, str]]) -> None:
    """签名里的下划线符号若只在英文散文中被拆开，则记为误译。"""
    signature = record["signature"]
    english = " ".join(_text(record.get(name), "en") for name in ("title", "summary", "description"))
    for ident in _IDENTIFIERS.findall(signature):
        spaced = ident.replace("_", " ")
        if spaced in english and ident not in english:
            detected.append({"id": record["id"], "code": "symbol_translated", "field": "summary", "detail": ident})
            return


def _text(value: object, lang: str) -> str:
    """读取一个中英字段；缺失按空字符串处理。"""
    if not isinstance(value, dict):
        return ""
    item = value.get(lang, "")
    return item if isinstance(item, str) else ""


def _norm(text: str) -> str:
    """去掉大小写和标点后再比较标题与摘要。"""
    return "".join(char.lower() for char in text if char.isalnum())
