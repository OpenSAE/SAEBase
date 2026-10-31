"""Source-owned MMLUPro QA adapter (independent of any experiment)."""

from __future__ import annotations
from string import ascii_uppercase

from typing import Any

from saebase.data.qa_records import DatasetSpec, text, choices

SPEC = DatasetSpec(
    "mmlu_pro",
    "TIGER-Lab/MMLU-Pro",
    None,
    "test",
    "mmlu",
    False,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    raw = list(row.get("options") or [])
    key = row.get("answer_index")
    if key is None:
        key = ascii_uppercase.index(text(row.get("answer")).upper())
    options, answer = choices(
        [(ascii_uppercase[i], option) for i, option in enumerate(raw)],
        ascii_uppercase[int(key)],
    )
    return {
        "question": text(row.get("question")),
        "answer": answer,
        "choices": options,
        "answer_type": "choice",
        "category": text(row.get("category") or source_config or "all"),
    }
