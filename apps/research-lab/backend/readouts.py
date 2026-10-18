"""CPU-only task readout and aligned-input contracts.

No model inference takes place here. A GPU worker must supply the actual
per-token teacher-forced log probabilities, and must never put gold answers
inside the *base* activation prompt. This module validates and aggregates
saved observations without inventing a model score.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CandidateEvidence:
    """One complete candidate continuation, not a first-token proxy."""

    label: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Candidate label is missing")
        if not self.token_ids or len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("Each candidate requires a complete nonempty logprob sequence")
        if any(type(value) is not int or value < 0 for value in self.token_ids):
            raise ValueError("Invalid token IDs")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value > 1e-8 for value in self.token_logprobs):
            raise ValueError("Per-token log probabilities must be finite and nonpositive")


def score_continuations(
    observations: Sequence[CandidateEvidence],
    *,
    aggregation: str = "sum",
    gold_label: str | None = None,
) -> dict[str, Any]:
    """Rank complete continuations by measured total/mean log probability.

    For benchmarking, the scoring policy is part of the protocol identity.
    Mean-normalized and summed continuation scores are *not interchangeable*.
    Do not substitute only the next-token logits for a multi-token candidate.
    """
    if aggregation not in ("sum", "mean"):
        raise ValueError("Allowed score aggregation: sum or mean")
    if not observations:
        raise ValueError("At least one measured continuation is needed")
    labels = [item.label for item in observations]
    if len(labels) != len(set(labels)):
        raise ValueError("Candidate labels must be unique")
    if gold_label is not None and gold_label not in labels:
        raise ValueError("Gold label not among candidate choices")
    scores = []
    for item in observations:
        total = math.fsum(item.token_logprobs)
        score = total if aggregation == "sum" else total / len(item.token_ids)
        scores.append(
            {
                "label": item.label,
                "score": score,
                "sum_log_prob": total,
                "mean_log_prob": total / len(item.token_ids),
                "n_tokens": len(item.token_ids),
            }
        )
    best = max(row["score"] for row in scores)
    winners = [row["label"] for row in scores if math.isclose(row["score"], best, rel_tol=1e-12, abs_tol=1e-12)]
    return {
        "schema": "opensae.task_readout.v1",
        "source": "measured_per_token_logprobs",
        "aggregation": aggregation,
        "scores": scores,
        "predicted_label": winners[0] if len(winners) == 1 else None,
        "tie": len(winners) > 1,
        "gold_label": gold_label,
        "correct": (winners[0] == gold_label if len(winners) == 1 else False) if gold_label is not None else None,
        "interpretation": "complete teacher-forced candidate log probabilities; not free-generation accuracy",
    }


def input_anchor(
    base_text: str,
    variant_text: str,
    *,
    anchor: str,
    offset_mapping: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Locate a preserved, unique substring to prevent misaligned RSS reads.

    A tokenizer offset mapping refers to the *variant* tokenization. The
    corresponding base text needs its own independently computed mapping.
    """
    if not all(isinstance(x, str) for x in (base_text, variant_text, anchor)) or not anchor:
        raise ValueError("Invalid text anchor")
    if base_text.count(anchor) != 1 or variant_text.count(anchor) != 1:
        raise ValueError("Anchor must occur exactly once in base and variant")
    base_start = base_text.index(anchor)
    variant_start = variant_text.index(anchor)
    end = variant_start + len(anchor)
    token_positions = None
    if offset_mapping is not None:
        accepted: list[int] = []
        for index, pair in enumerate(offset_mapping):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or any(type(p) is not int for p in pair)
            ):
                raise ValueError("Malformed tokenizer offsets")
            start, stop = pair
            if start < 0 or stop < start or stop > len(variant_text):
                raise ValueError("Tokenizer offset lies outside transformed input")
            if start < end and stop > variant_start:
                if start < variant_start or stop > end:
                    raise ValueError("Anchor cuts through a tokenizer token")
                accepted.append(index)
        if not accepted or accepted != list(range(accepted[0], accepted[-1] + 1)):
            raise ValueError("No contiguous aligned tokenizer positions")
        token_positions = accepted
    return {
        "base_anchor_span": [base_start, base_start + len(anchor)],
        "variant_anchor_span": [variant_start, end],
        "token_positions": token_positions,
        "anchor_sha256": hashlib.sha256(anchor.encode("utf-8")).hexdigest(),
        "anchor_text_equal": True,
    }
