"""Source-owned BoolQ QA adapter (independent of any experiment)."""

from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text

SPEC = DatasetSpec(
    "boolq",
    "google/boolq",
    None,
    "validation",
    "boolq",
    False,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    if type(row.get("answer")) is not bool:
        raise ValueError("BoolQ reference answer must be a boolean")
    return {
        "question": text(row.get("question")),
        "answer": "Yes" if row["answer"] else "No",
        "context": text(row.get("passage")),
        "answer_type": "boolean",
    }
