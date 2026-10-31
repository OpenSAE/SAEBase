"""Source-owned GSM8K QA adapter (independent of any experiment)."""

from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text

SPEC = DatasetSpec(
    "gsm8k",
    "openai/gsm8k",
    "main",
    "test",
    "gsm8k",
    False,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    solution = text(row.get("answer"))
    if "####" not in solution:
        raise ValueError("GSM8K final-answer delimiter missing")
    return {
        "question": text(row.get("question")),
        "answer": solution.rsplit("####", 1)[1].strip().replace(",", ""),
        "answer_type": "number",
    }
