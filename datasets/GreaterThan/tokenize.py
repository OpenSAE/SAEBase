"""Tokenization and filtering for GreaterThan prompts."""

from __future__ import annotations

from typing import Any, Literal

Strategy = Literal["single_token_yy", "first_token_yy"]


def _build_two_digit_token_map(model: Any) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for yy in range(100):
        toks = model.to_tokens(f"{yy:02d}", prepend_bos=False)
        if toks.shape[1] == 1:
            mapping[yy] = int(toks[0, 0].item())
    return mapping


def attach_token_ids_single_token_yy(
    items: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    digit_map = _build_two_digit_token_map(model)
    kept: list[dict[str, Any]] = []
    for ex in items:
        yy = int(ex["YY"])
        greater_ids = [digit_map[v] for v in range(yy + 1, 100) if v in digit_map]
        leq_ids = [digit_map[v] for v in range(0, yy + 1) if v in digit_map]
        if len(greater_ids) < 5 or len(leq_ids) < 5:
            continue
        kept.append(
            {
                "id": f"greater_than-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "century": int(ex["century"]),
                    "YY": yy,
                },
                "tokenization": {
                    "greater_ids": greater_ids,
                    "less_equal_ids": leq_ids,
                    "strategy": "single_token_yy",
                },
                "century": int(ex["century"]),
                "YY": yy,
                "greater_tok_ids": greater_ids,
                "less_equal_tok_ids": leq_ids,
            }
        )
    return kept


def attach_token_ids_first_token_yy(
    items: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    """Gemma-style: partition by first token of each YY suffix (may be multi-token)."""
    kept: list[dict[str, Any]] = []
    for ex in items:
        yy = int(ex["YY"])
        greater: set[int] = set()
        leq: set[int] = set()
        for v in range(100):
            toks = model.to_tokens(f"{v:02d}", prepend_bos=False)
            tid = int(toks[0, 0].item())
            if v >= yy + 1:
                greater.add(tid)
            else:
                leq.add(tid)
        if len(greater) < 3 or len(leq) < 3:
            continue
        kept.append(
            {
                "id": f"greater_than-{len(kept):06d}",
                "prompt": ex["prompt"],
                "target": {
                    "century": int(ex["century"]),
                    "YY": yy,
                },
                "tokenization": {
                    "greater_ids": sorted(greater),
                    "less_equal_ids": sorted(leq),
                    "strategy": "first_token_yy",
                },
                "century": int(ex["century"]),
                "YY": yy,
                "greater_tok_ids": sorted(greater),
                "less_equal_tok_ids": sorted(leq),
            }
        )
    return kept


def attach_token_ids(
    items: list[dict[str, Any]],
    model: Any,
    *,
    strategy: Strategy = "single_token_yy",
) -> list[dict[str, Any]]:
    if strategy == "single_token_yy":
        return attach_token_ids_single_token_yy(items, model)
    if strategy == "first_token_yy":
        return attach_token_ids_first_token_yy(items, model)
    raise ValueError(f"Unknown GreaterThan tokenize strategy: {strategy!r}")


def filter_greedy_in_greater(
    items: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        pred = int(logits.argmax().item())
        greater_set = set(int(t) for t in ex["greater_tok_ids"])
        if pred in greater_set:
            kept.append(ex)
    return kept


def strategy_for_model_key(model_key: str) -> Strategy:
    key = str(model_key)
    if key.startswith("gemma"):
        return "first_token_yy"
    return "single_token_yy"
