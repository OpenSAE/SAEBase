"""Source-owned GPQAExtended QA adapter (independent of any experiment)."""

from __future__ import annotations
import random

from typing import Any

from saebase.data.qa_records import DatasetSpec, text, choices, digest

SPEC = DatasetSpec(
    "gpqa_extended",
    "Idavidrein/gpqa",
    "gpqa_extended",
    "train",
    "gpqa",
    True,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    question = text(row.get("Question"))
    correct = text(row.get("Correct Answer"))
    others = [text(row.get(f"Incorrect Answer {i}")) for i in (1, 2, 3)]
    if not correct or any(not other for other in others):
        raise ValueError("GPQA answer options missing")
    values = [correct, *others]
    random.Random(int(digest([seed, question])[:12], 16)).shuffle(values)
    options, answer = choices(
        [(str(i), value) for i, value in enumerate(values)],
        str(values.index(correct)),
    )
    return {
        "question": question,
        "answer": answer,
        "choices": options,
        "answer_type": "choice",
        "category": text(row.get("High-level domain") or row.get("Subdomain") or "science"),
    }
