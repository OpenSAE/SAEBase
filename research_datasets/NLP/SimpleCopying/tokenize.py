"""Tokenization and filtering for SimpleCopying prompts."""

from __future__ import annotations

from typing import Any


def attach_token_ids(items: list[dict[str, str]], model: Any) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ex in items:
        t_c = model.to_tokens(" " + ex["correct_name"], prepend_bos=False)
        t_d = model.to_tokens(" " + ex["distractor_name"], prepend_bos=False)
        if t_c.shape[1] != 1 or t_d.shape[1] != 1:
            continue
        kept.append(
            {
                "id": f"simple_copying-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "correct_text": ex["correct_name"],
                    "distractor_text": ex["distractor_name"],
                    "name_a": ex["name_a"],
                    "name_b": ex["name_b"],
                },
                "tokenization": {
                    "input_ids": [int(x) for x in model.to_tokens(ex["prompt"])[0].tolist()],
                    "correct_ids": [int(t_c[0, 0].item())],
                    "distractor_ids": [int(t_d[0, 0].item())],
                },
                "tok_correct": int(t_c[0, 0].item()),
                "tok_distractor": int(t_d[0, 0].item()),
                "correct_name": ex["correct_name"],
                "distractor_name": ex["distractor_name"],
                "name_a": ex["name_a"],
                "name_b": ex["name_b"],
            }
        )
    return kept


def filter_logit_diff(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    """Keep rows where logit(correct) > logit(distractor) at last position."""
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        if float(logits[int(ex["tok_correct"])]) > float(logits[int(ex["tok_distractor"])]):
            kept.append(ex)
    return kept


# Backwards-compatible alias
filter_with_model = filter_logit_diff
