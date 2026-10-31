"""Source-owned MedQA QA adapter (independent of any experiment)."""

from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text, choices

SPEC = DatasetSpec(
    "medqa",
    "GBaker/MedQA-USMLE-4-options",
    None,
    "test",
    "medqa",
    False,
)


def parse(row: dict[str, Any], *, source_config: str | None, seed: int) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    opts = row.get("options") or {}
    if not isinstance(opts, dict):
        raise ValueError("MedQA options must be a label-to-text dictionary")
    options, answer = choices(sorted(opts.items()), text(row.get("answer_idx")))
    return {
        "question": text(row.get("question")),
        "answer": answer,
        "choices": options,
        "answer_type": "choice",
    }
