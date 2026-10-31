"""Source-owned CommonsenseQA QA adapter (independent of any experiment)."""
from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text, choices, digest

SPEC = DatasetSpec(
    "commonsense_qa", "tau/commonsense_qa",
    None, "validation",
    "commonsense", False,
)


def parse(
    row: dict[str, Any], *, source_config: str | None, seed: int
) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    values = row.get("choices") or {}
    options, answer = choices(
        zip(values.get("label") or [], values.get("text") or []),
        text(row.get("answerKey")),
    )
    return {"question": text(row.get("question")), "answer": answer,
            "choices": options, "answer_type": "choice"}
