"""Tokenization and filtering for SubjectVerbAgreement prompts."""

from __future__ import annotations

from typing import Any


def attach_token_ids(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ex in items:
        t_c = model.to_tokens(" " + ex["correct_verb"], prepend_bos=False)
        t_d = model.to_tokens(" " + ex["distractor_verb"], prepend_bos=False)
        if t_c.shape[1] != 1 or t_d.shape[1] != 1:
            continue
        kept.append(
            {
                "id": f"sva-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "correct_text": ex["correct_verb"],
                    "distractor_text": ex["distractor_verb"],
                    "subj_is_plural": bool(ex["subj_is_plural"]),
                },
                "tokenization": {
                    "correct_ids": [int(t_c[0, 0].item())],
                    "distractor_ids": [int(t_d[0, 0].item())],
                },
                "tok_correct": int(t_c[0, 0].item()),
                "tok_distractor": int(t_d[0, 0].item()),
                "correct_verb": ex["correct_verb"],
                "distractor_verb": ex["distractor_verb"],
                "subj_is_plural": bool(ex["subj_is_plural"]),
            }
        )
    return kept


def filter_argmax(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        if int(logits.argmax().item()) == int(ex["tok_correct"]):
            kept.append(ex)
    return kept


def filter_logit_diff(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        if float(logits[int(ex["tok_correct"])]) > float(logits[int(ex["tok_distractor"])]):
            kept.append(ex)
    return kept


def filter_with_model(
    items: list[dict[str, Any]],
    model: Any,
    *,
    min_kept: int = 100,
) -> tuple[list[dict[str, Any]], str]:
    kept = filter_argmax(items, model)
    policy = "argmax_correct"
    if len(kept) < min_kept:
        kept = filter_logit_diff(items, model)
        policy = "logit_diff"
    return kept, policy
