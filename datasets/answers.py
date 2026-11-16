"""Dataset-owned canonical answers for answer-conditioned scientific readouts.

Preserves original prompts and labels. Explicit source fields define the answer;
absence of a reference response is not filled by model predictions or safety labels.
"""

from __future__ import annotations

import re


class ReferenceAnswerUnavailable(ValueError):
    """The selected record has no supported canonical answer text."""


def reference_answer(row, dataset):
    raw = row.get("source_record", {})
    raw = raw if isinstance(raw, dict) else {}
    prefix = " "
    if isinstance(row.get("reference_answer"), (str, int, float)):
        text, source = str(row["reference_answer"]), "reference_answer"
        if dataset == "RAVEL" and row.get("prompt", "").endswith(('"', "'")):
            prefix = ""
    elif isinstance(row.get("correct_answer"), str):
        text, source = row["correct_answer"], "correct_answer"
    elif dataset == "IOI":
        text, source = row.get("target", {}).get("correct_text"), "target.correct_text"
    elif dataset == "CounterFact":
        text = raw.get("requested_rewrite", {}).get("target_true", {}).get("str")
        source = "source_record.requested_rewrite.target_true.str"
    elif dataset in {"HotpotQA", "GSM8K"}:
        text, source = raw.get("answer"), "source_record.answer"
        if dataset == "GSM8K" and isinstance(text, str) and "####" in text:
            text = text.rsplit("####", 1)[1]
            source += " (final answer after ####)"
    elif dataset == "BBH":
        text, source = raw.get("target"), "source_record.target"
        if isinstance(text, str) and re.fullmatch(r"\([A-Z]\)", text.strip()):
            choices = re.findall(
                r"(?m)^\(" + re.escape(text.strip()[1]) + r"\)\s*(.+)$", raw.get("input", "")
            )
            if len(choices) == 1:
                text = choices[0]
                source += " resolved to the labeled option text in source_record.input"
    elif dataset == "ARCChallenge":
        choices = raw.get("choices", {})
        labels, texts = choices.get("label", []), choices.get("text", [])
        matches = [
            value
            for label, value in zip(labels, texts, strict=True)
            if label == raw.get("answerKey")
        ]
        text, source = (
            (matches[0] if len(matches) == 1 else None),
            "source_record.choices.text[answerKey]",
        )
    elif dataset == "MMLUPro":
        options, index = raw.get("options", []), raw.get("answer_index")
        if type(index) is int and 0 <= index < len(options):
            if raw.get("answer") != chr(65 + index):
                raise ValueError("MMLUPro answer letter and answer_index disagree")
            text = options[index]
        else:
            text = None
        source = "source_record.options[answer_index]"
    elif dataset == "TruthfulQA":
        choices = raw.get("mc1_targets", {})
        correct = [
            value
            for value, label in zip(
                choices.get("choices", []), choices.get("labels", []), strict=True
            )
            if label == 1
        ]
        text = correct[0] if correct else None
        source = "first labeled correct source_record.mc1_targets choice in original order"
    elif dataset == "HumanEval":
        text, source = raw.get("canonical_solution"), "source_record.canonical_solution"
        prefix = "" if row.get("prompt", "").endswith("\n") else "\n"
    else:
        raise ReferenceAnswerUnavailable(
            f"{dataset} has no canonical answer response in this record; supply reference_answer explicitly"
        )
    if not isinstance(text, str) or not text.strip():
        raise ReferenceAnswerUnavailable(f"{dataset}: canonical answer missing at {source}")
    if dataset != "HumanEval":
        text = text.strip()
    return {
        "text": text,
        "prefix": prefix,
        "source_field": source,
        "dataset": dataset,
        "prompt_unchanged": True,
    }
