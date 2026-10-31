"""Source-owned HotpotQA QA adapter (independent of any experiment)."""

from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text

SPEC = DatasetSpec(
    "hotpot_qa",
    "hotpotqa/hotpot_qa",
    "distractor",
    "validation",
    "hotpot",
    False,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    ctx = row.get("context") or {}
    titles, sentences = ctx.get("title") or [], ctx.get("sentences") or []
    if len(titles) != len(sentences):
        raise ValueError("Hotpot context title/sentence arrays are misaligned")
    context = "\n".join(
        f"{text(title)}: {' '.join(map(str, block))}" for title, block in zip(titles, sentences)
    )
    return {
        "question": text(row.get("question")),
        "answer": text(row.get("answer")),
        "context": context,
        "answer_type": "exact_text",
    }
