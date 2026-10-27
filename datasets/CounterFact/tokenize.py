"""Tokenization and filtering for CounterFact prompts."""

from __future__ import annotations

from typing import Any


def attach_and_filter_argmax(
    items: list[dict[str, str]],
    model: Any,
    *,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Keep rows where greedy last-token pred matches target_true (single token)."""
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    subset = items if max_items is None else items[:max_items]
    for ex in subset:
        toks_true = model.to_tokens(" " + ex["target_true"].strip(), prepend_bos=False)
        toks_false = model.to_tokens(" " + ex["target_false"].strip(), prepend_bos=False)
        if toks_true.shape[1] != 1 or toks_false.shape[1] != 1:
            continue
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        pred = int(logits.argmax().item())
        t_true = int(toks_true[0, 0].item())
        t_false = int(toks_false[0, 0].item())
        if pred != t_true:
            continue
        kept.append(
            {
                "id": f"counterfact-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "correct_text": ex["target_true"],
                    "distractor_text": ex["target_false"],
                },
                "tokenization": {
                    "correct_ids": [t_true],
                    "distractor_ids": [t_false],
                },
                "target_true": ex["target_true"],
                "target_false": ex["target_false"],
                "tok_true": t_true,
                "tok_false": t_false,
            }
        )
    return kept


def attach_and_filter_logit_diff(
    items: list[dict[str, str]],
    model: Any,
    *,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    subset = items if max_items is None else items[:max_items]
    for ex in subset:
        toks_true = model.to_tokens(" " + ex["target_true"].strip(), prepend_bos=False)
        toks_false = model.to_tokens(" " + ex["target_false"].strip(), prepend_bos=False)
        if toks_true.shape[1] != 1 or toks_false.shape[1] != 1:
            continue
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        t_true = int(toks_true[0, 0].item())
        t_false = int(toks_false[0, 0].item())
        if float(logits[t_true]) <= float(logits[t_false]):
            continue
        kept.append(
            {
                "id": f"counterfact-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "correct_text": ex["target_true"],
                    "distractor_text": ex["target_false"],
                },
                "tokenization": {
                    "correct_ids": [t_true],
                    "distractor_ids": [t_false],
                },
                "target_true": ex["target_true"],
                "target_false": ex["target_false"],
                "tok_true": t_true,
                "tok_false": t_false,
            }
        )
    return kept


def filter_with_model(
    items: list[dict[str, str]],
    model: Any,
    *,
    min_kept: int = 100,
    max_items: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    kept = attach_and_filter_argmax(items, model, max_items=max_items)
    policy = "argmax_correct"
    if len(kept) < min_kept:
        kept = attach_and_filter_logit_diff(items, model, max_items=max_items)
        policy = "logit_diff"
    return kept, policy
