"""审计测验用中英覆盖层。缺中文和缺英文都算问题，函数不补译文。"""

from __future__ import annotations

import re


class LocaleRowError(ValueError):
    """覆盖层记录无法检查。"""


_PLACEHOLDER = "english translation unavailable"
_TEMPLATES = (
    "this function implements the requested behavior.",
    "this module handles the feature.",
)
_IDENTIFIERS = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b")


def review_locale_rows(rows: list[dict]) -> dict:
    """按稳定 ID 检查中英字段、占位文案、乱码、项目 ID 和被拆开的符号。

    detected 只列出问题。fixed 始终为空，因为这里不写回翻译。
    """
    if not isinstance(rows, list):
        raise LocaleRowError("rows must be a list")
    detected: list[dict[str, str]] = []
    preserved: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("synthetic") is not True:
            raise LocaleRowError("each row must be marked synthetic")
        node_id = row.get("id")
        signature = row.get("signature")
        path = row.get("path")
        if not all(isinstance(item, str) and item for item in (node_id, signature, path)):
            raise LocaleRowError("id, signature, and path must be nonempty strings")
        if node_id in seen:
            raise LocaleRowError(f"duplicate id: {node_id}")
        seen.add(str(node_id))
        preserved.append({"id": str(node_id), "signature": str(signature), "path": str(path)})
        _one_row(row, detected)
    detected.sort(key=lambda item: (item["id"], item["code"], item["field"]))
    return {
        "detected": detected,
        "fixed": [],
        "preserved": preserved,
        "note": "detected lists problems only; nothing in the authority translation tables was edited",
    }


def _one_row(row: dict, detected: list[dict[str, str]]) -> None:
    """检查一条记录的项目、签名和中英文本。"""
    node_id = str(row["id"])
    if row.get("project") != "SAEBase":
        detected.append({"id": node_id, "code": "wrong_project", "field": "project", "detail": str(row.get("project"))})
    localized = row.get("localized_signature")
    if isinstance(localized, str) and localized != row["signature"]:
        detected.append({"id": node_id, "code": "symbol_translated", "field": "localized_signature", "detail": localized})
    for name in ("title", "summary", "description"):
        _language_field(row, name, detected)
    for lang, code in (("en", "title_repeats_summary"), ("zh", "title_repeats_summary")):
        title = _phrase(row.get("title"), lang)
        summary = _phrase(row.get("summary"), lang)
        if title and summary and _compact(title) == _compact(summary):
            detected.append({"id": node_id, "code": code, "field": f"summary.{lang}", "detail": summary})
    _spaced_symbols(row, detected)


def _language_field(row: dict, name: str, detected: list[dict[str, str]]) -> None:
    """中文或英文缺失、占位句和乱码各自记一条。"""
    value = row.get(name)
    for lang, code in (("zh", "missing_zh"), ("en", "missing_en")):
        text = _phrase(value, lang)
        field = f"{name}.{lang}"
        if text == "":
            detected.append({"id": str(row["id"]), "code": code, "field": field, "detail": "empty"})
            continue
        if lang == "en" and (_PLACEHOLDER in text.lower() or _compact(text) in {_compact(item) for item in _TEMPLATES}):
            detected.append({"id": str(row["id"]), "code": "translation_placeholder", "field": field, "detail": text})
        if "\ufffd" in text:
            detected.append({"id": str(row["id"]), "code": "undecodable_text", "field": field, "detail": text})


def _spaced_symbols(row: dict, detected: list[dict[str, str]]) -> None:
    """签名里的下划线符号若只在英文散文中被拆开，则记为误译。"""
    signature = str(row["signature"])
    english = " ".join(_phrase(row.get(name), "en") for name in ("title", "summary", "description"))
    for ident in _IDENTIFIERS.findall(signature):
        spaced = ident.replace("_", " ")
        if spaced in english and ident not in english:
            detected.append({"id": str(row["id"]), "code": "symbol_translated", "field": "summary.en", "detail": ident})
            return


def _phrase(value: object, lang: str) -> str:
    """读取一个中英字段。缺失时返回空字符串，不生成译文。"""
    if not isinstance(value, dict):
        return ""
    item = value.get(lang, "")
    return item if isinstance(item, str) else ""


def _compact(text: str) -> str:
    """去掉大小写和标点，用来识别只多了一个句号的复述。"""
    return "".join(char.lower() for char in text if char.isalnum())
