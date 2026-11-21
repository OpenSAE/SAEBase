"""Tokenization filters for MMLU-Pro answer letters (no greedy correctness filter)."""

from __future__ import annotations

from typing import Any


def attach_token_ids(
    items: list[dict[str, Any]],
    model: Any,
    *,
    letter_prefix: str = " ",
    max_prompt_tokens: int | None = 2048,
) -> list[dict[str, Any]]:
    """Keep rows where answer/distractor letters are single tokens and prompt fits."""
    kept: list[dict[str, Any]] = []
    for ex in items:
        prompt = str(ex["prompt"])
        prompt_toks = model.to_tokens(prompt)
        n_tok = int(prompt_toks.shape[-1])
        if max_prompt_tokens is not None and n_tok > int(max_prompt_tokens):
            continue
        t_c = model.to_tokens(letter_prefix + str(ex["answer"]), prepend_bos=False)
        t_d = model.to_tokens(letter_prefix + str(ex["distractor"]), prepend_bos=False)
        if t_c.shape[1] != 1 or t_d.shape[1] != 1:
            continue
        tok_correct = int(t_c[0, 0].item())
        tok_distractor = int(t_d[0, 0].item())
        qid = ex.get("question_id", len(kept))
        kept.append(
            {
                "id": f"mmlu_pro-{int(qid):06d}",
                "prompt": prompt,
                "question_id": qid,
                "category": ex.get("category"),
                "src": ex.get("src"),
                "options": list(ex.get("options") or []),
                "answer": ex["answer"],
                "answer_index": int(ex["answer_index"]),
                "distractor": ex["distractor"],
                "target": {
                    "correct_text": str(ex["answer"]),
                    "distractor_text": str(ex["distractor"]),
                    "category": ex.get("category"),
                },
                "tokenization": {
                    "correct_ids": [tok_correct],
                    "distractor_ids": [tok_distractor],
                    "n_prompt_tokens": n_tok,
                },
                "tok_correct": tok_correct,
                "tok_distractor": tok_distractor,
            }
        )
    return kept
