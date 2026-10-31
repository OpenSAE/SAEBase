"""Source-owned BBH QA adapter (independent of any experiment)."""
from __future__ import annotations

from typing import Any

from saebase.data.qa_records import DatasetSpec, text, choices, digest

SPEC = DatasetSpec(
    "bbh", "lukaemon/bbh",
    None, "test",
    "bbh", False,
)


def parse(
    row: dict[str, Any], *, source_config: str | None, seed: int
) -> dict[str, Any]:
    """Extract real QA fields; select/score samples in the owning experiment."""
    return {"question": text(row.get("input")),
            "answer": text(row.get("target")), "answer_type": "exact_text",
            "category": text(source_config or "all")}
