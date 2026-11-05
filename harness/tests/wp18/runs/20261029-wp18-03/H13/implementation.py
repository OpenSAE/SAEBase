"""Report missing zh or en fields without inventing a translation."""

from __future__ import annotations

_FIELDS = ("title", "summary", "description")
_LANGS = ("zh", "en")


class InputError(ValueError):
    """Raised for duplicate ids, a foreign project, or a record that is not bilingual metadata."""


def solve(payload: dict) -> dict:
    """Collect missing-language defects and leave technical identity untouched."""
    if not isinstance(payload, dict) or set(payload) != {"records"}:
        raise InputError("schema")
    records = payload["records"]
    if not isinstance(records, list):
        raise InputError("records")
    defects: list[str] = []
    seen: set[str] = set()
    for record in records:
        defects.extend(_record(record, seen))
    return {"defects": defects}


def _record(record: object, seen: set[str]) -> list[str]:
    """Read one record. A missing language is a defect; a bad project is an error."""
    if not isinstance(record, dict):
        raise InputError("record")
    allowed = {"id", "title", "summary", "description", "project"}
    if set(record) - allowed or not {"id", "title", "summary", "description"} <= set(record):
        raise InputError("record")
    if "project" in record and record["project"] != "SAEBase":
        raise InputError("project")
    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id or record_id in seen:
        raise InputError("id")
    seen.add(record_id)
    defects: list[str] = []
    for field in _FIELDS:
        block = record[field]
        if not isinstance(block, dict):
            raise InputError(field)
        for lang in _LANGS:
            text = block.get(lang)
            if not isinstance(text, str) or not text.strip():
                defects.append(f"{record_id}:{field}.{lang}:missing")
                continue
            if "\ufffd" in text or text.casefold().strip() == "english translation unavailable":
                raise InputError("text")
    for lang in _LANGS:
        title = record["title"].get(lang)
        summary = record["summary"].get(lang)
        if isinstance(title, str) and isinstance(summary, str) and _same_purpose(title, summary):
            raise InputError("summary")
    return defects


def _same_purpose(title: str, summary: str) -> bool:
    """Treat punctuation-only differences as the same purpose."""
    title_key = "".join(ch for ch in title.casefold() if ch.isalnum())
    summary_key = "".join(ch for ch in summary.casefold() if ch.isalnum())
    return bool(title_key) and title_key == summary_key
