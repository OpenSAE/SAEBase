"""Dataset-independent QA record identities and zero-shot rendering.

Adapters own source-specific schemas. Experiments select subsets and tasks.
This module knows no dataset list, SAE model matrix, or experiment budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from string import ascii_uppercase
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo: str
    config: str | None
    split: str
    kind: str
    restricted: bool = False


def digest(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def choices(pairs: Iterable[tuple[str, Any]], gold_key: str) -> tuple[list[dict[str, str]], str]:
    options: list[dict[str, str]] = []
    canonical_gold = ""
    for index, (label, option) in enumerate(pairs):
        if index >= 16:
            raise ValueError("Too many multiple-choice options")
        content = text(option)
        if not content or content == "N/A":
            continue
        new_label = ascii_uppercase[len(options)]
        options.append({"label": new_label, "text": content})
        if text(label).upper() == text(gold_key).upper():
            canonical_gold = new_label
    if len(options) < 2 or canonical_gold not in {o["label"] for o in options}:
        raise ValueError("Invalid choices or reference answer")
    return options, canonical_gold


def render_prompt(record: dict[str, Any]) -> str:
    kind = record["answer_type"]
    context = text(record.get("context"))
    header = (
        "Select the correct answer. Respond with only the option letter.\n"
        if kind == "choice"
        else "Answer the question. Give only the final answer.\n"
    )
    parts = [header]
    if context:
        parts.append(f"Context:\n{context}\n")
    parts.append(f"Question:\n{record['question']}\n")
    if kind == "choice":
        parts.extend(f"{c['label']}. {c['text']}\n" for c in record["choices"])
        parts.append("Answer:")
    elif kind == "boolean":
        parts.append("Answer (Yes or No):")
    else:
        parts.append("Final answer:")
    return "".join(parts)


def make_record(
    spec: DatasetSpec,
    source: dict[str, Any],
    fields: dict[str, Any],
    source_config: str | None,
    source_index: int,
) -> dict[str, Any]:
    """Attach generic identity/provenance to a source-specific parsed row."""
    question = text(fields.get("question"))
    answer = text(fields.get("answer"))
    if not question or not answer:
        raise ValueError("Question or reference answer is empty")
    source_id = text(
        source.get("question_id")
        or source.get("id")
        or source.get("_id")
        or source.get("Record ID")
        or source_index
    )
    sample_id = digest([spec.key, source_config, spec.split, source_id, question])[:24]
    record = {
        "sample_id": sample_id,
        "dataset_id": spec.key,
        "source_repo": spec.repo,
        "source_config": source_config,
        "source_split": spec.split,
        "source_id": source_id,
        "question": question,
        "context": text(fields.get("context")),
        "choices": list(fields.get("choices") or []),
        "answer": answer,
        "answer_type": fields.get("answer_type", "exact_text"),
        "category": text(
            fields.get("category")
            or source.get("category")
            or source.get("type")
            or source_config
            or "all"
        ),
        "prompt_group_id": sample_id,
        "restricted_source": spec.restricted,
    }
    prompt = render_prompt(record)
    if len(prompt) > 12000:
        raise ValueError("Prompt exceeds frozen 12,000-character ceiling")
    record["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return record
