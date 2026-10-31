"""Source-owned MATH500 QA adapter (independent of any experiment)."""

from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text

SPEC = DatasetSpec(
    "math500",
    "HuggingFaceH4/MATH-500",
    None,
    "test",
    "math500",
    False,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    return {
        "question": text(row.get("problem")),
        "answer": text(row.get("answer")),
        "answer_type": "math",
    }
