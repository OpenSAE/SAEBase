"""Tokenization and filtering for IOI prompts."""

from __future__ import annotations

from typing import Any


def attach_token_ids(items: list[dict[str, str]], model: Any) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ex in items:
        t_io = model.to_tokens(" " + ex["io_name"], prepend_bos=False)
        t_s = model.to_tokens(" " + ex["s_name"], prepend_bos=False)
        if t_io.shape[1] != 1 or t_s.shape[1] != 1:
            continue
        kept.append(
            {
                "id": f"ioi-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "correct_text": ex["io_name"],
                    "distractor_text": ex["s_name"],
                },
                "tokenization": {
                    "correct_ids": [int(t_io[0, 0].item())],
                    "distractor_ids": [int(t_s[0, 0].item())],
                },
                "io_name": ex["io_name"],
                "s_name": ex["s_name"],
                "tok_io": int(t_io[0, 0].item()),
                "tok_s": int(t_s[0, 0].item()),
            }
        )
    return kept


def filter_argmax(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        if int(logits.argmax().item()) == int(ex["tok_io"]):
            kept.append(ex)
    return kept


def filter_with_model(
    items: list[dict[str, Any]],
    model: Any,
    *,
    min_kept: int = 100,
) -> tuple[list[dict[str, Any]], str]:
    kept = filter_argmax(items, model)
    return kept, "argmax_correct"
