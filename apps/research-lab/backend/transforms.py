"""Deterministic proposed RSS prompt variants; never assumed meaning-preserving.

Each supported base prompt creates 3 settings x 5 transformation operators x
3 realizations = 45 candidate variants. Exactly one realization per operator
belongs to selection and the other two to held-out. These are separate
*measurements of one parent prompt*, not 45 independent prompt groups.

No variant becomes a public validated experiment without semantic review,
tokenizer-specific anchor checks, source approval and actual SAE activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .core import canonical_json, dataset_catalog
from .ingest import _atomic_bytes
from .readouts import input_anchor

SCHEMA = "opensae.rss.variant_proposals.v1"
SETTINGS = ("local_format", "instruction_wrapper", "context_wrapper")
NOT_AUTOMATED = {"wikitext103", "tinystories", "humaneval"}

PREFIX_SETS = (
    (
        "Answer the following request as written:\n",
        "Please respond to the request below:\n",
        "Read the request and answer it:\n",
    ),
    (
        "Follow the supplied task carefully:\n",
        "Consider the task in the following section:\n",
        "Work from the given task statement:\n",
    ),
    (
        "Provide a direct response to the following:\n",
        "Give a straightforward answer to this:\n",
        "Answer this standalone request:\n",
    ),
    (
        "The user's original task is shown below:\n",
        "The question to be answered appears below:\n",
        "Here is the task text to consider:\n",
    ),
    (
        "Consider the complete question that follows:\n",
        "Use the stated question and its information:\n",
        "Address the question below:\n",
    ),
)
CONTEXT_SETS = (
    (
        "The following is a sample from an evaluation document.\n",
        "The next item is part of a written evaluation collection.\n",
        "This example comes from a task collection.\n",
    ),
    (
        "The surrounding note is informational only.\n",
        "This prefatory text contains no additional task facts.\n",
        "No new facts are supplied by this introduction.\n",
    ),
    (
        "One independent item follows.\n",
        "This is a standalone example.\n",
        "The next request stands on its own.\n",
    ),
    (
        "The material below is written in English.\n",
        "The following request is given as text.\n",
        "The next item is an English-language passage.\n",
    ),
    (
        "This is a hypothetical test item.\n",
        "The text below is one evaluation item.\n",
        "A single task appears next.\n",
    ),
)


def _variant_text(base: str, setting: str, operator: int, realization: int) -> str:
    if setting == "local_format":
        pad = realization + 1
        if operator == 0:
            return "\n" * pad + base
        if operator == 1:
            return " " * pad + base
        if operator == 2:
            return base + "\n" * pad
        if operator == 3:
            return "\n" * pad + base + "\n" * pad
        if operator == 4:
            return "-" * (3 + realization) + "\n" + base
    if setting == "instruction_wrapper":
        return PREFIX_SETS[operator][realization] + "\n" + base
    if setting == "context_wrapper":
        return CONTEXT_SETS[operator][realization] + "\n" + base
    raise ValueError("Unsupported transformation setting or operator")


def build_variants(
    dataset_id: str,
    sample_id: str,
    prompt_group_id: str,
    base_prompt: str,
    *,
    anchor: str,
    source_revision: str,
) -> dict[str, Any]:
    """Generate reviewed-by-default-FALSE transformations with stable identity."""
    if dataset_id not in {item["id"] for item in dataset_catalog()["sources"]}:
        raise ValueError("Unknown source dataset")
    if dataset_id in NOT_AUTOMATED:
        raise ValueError("Dataset requires an independently designed transformation protocol")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(ch not in "0123456789abcdef" for ch in source_revision)
    ):
        raise ValueError("Exact pinned 40-character source revision is required")
    for label, value in (
        ("dataset_id", dataset_id),
        ("sample_id", sample_id),
        ("prompt_group_id", prompt_group_id),
        ("base_prompt", base_prompt),
        ("anchor", anchor),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing " + label)
    if base_prompt.count(anchor) != 1:
        raise ValueError("Anchor must appear once in the original base input")
    variants: list[dict[str, Any]] = []
    for setting in SETTINGS:
        for operator in range(5):
            for realization in range(3):
                text = _variant_text(base_prompt, setting, operator, realization)
                alignment = input_anchor(base_prompt, text, anchor=anchor)
                key = {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "setting": setting,
                    "operator": operator,
                    "realization": realization,
                    "protocol": SCHEMA,
                    "source_revision": source_revision,
                    "base_sha256": hashlib.sha256(base_prompt.encode("utf-8")).hexdigest(),
                }
                variant_id = hashlib.sha256(canonical_json(key)).hexdigest()[:24]
                variants.append(
                    {
                        "variant_id": variant_id,
                        "parent_sample_id": sample_id,
                        "prompt_group_id": prompt_group_id,
                        "setting": setting,
                        "operator": operator,
                        "realization": realization,
                        "variant_role": "selection" if realization == 0 else "held_out",
                        "text": text,
                        "anchor_spans": alignment,
                        "semantic_review": "required",
                        "tokenizer_alignment": "pending",
                    }
                )
    if len(variants) != 45 or len({r["variant_id"] for r in variants}) != 45:
        raise AssertionError("Transformation manifest must contain exactly 45 unique proposals")
    return {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "source_revision": source_revision,
        "sample_id": sample_id,
        "prompt_group_id": prompt_group_id,
        "base_prompt_sha256": hashlib.sha256(base_prompt.encode("utf-8")).hexdigest(),
        "anchor_sha256": hashlib.sha256(anchor.encode("utf-8")).hexdigest(),
        "n_settings": 3,
        "n_operators_per_setting": 5,
        "n_proposals": 45,
        "fully_reviewed": False,
        "interpretation": "proposed expression changes, not certified task equivalence",
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="One approved base prompt JSON record"
    )
    parser.add_argument(
        "--anchor", required=True, help="Unique source anchor, unchanged across variants"
    )
    parser.add_argument("--output", required=True, type=Path, help="New immutable output")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing proposal release")
    with args.input.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    result = build_variants(
        record["dataset_id"],
        record["sample_id"],
        record["prompt_group_id"],
        record["prompt"],
        anchor=args.anchor,
        source_revision=record["source_revision"],
    )
    _atomic_bytes(args.output, canonical_json(result))
    print(
        json.dumps(
            {"proposal_count": result["n_proposals"], "output": str(args.output)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
