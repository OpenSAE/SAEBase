"""GreaterThan prompt generation, token-id attachment, and JSON persistence."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

NOUNS = [
    "war",
    "drought",
    "famine",
    "riot",
    "expedition",
    "siege",
    "movement",
    "festival",
    "eruption",
    "trial",
    "rebellion",
    "tour",
    "revolution",
    "celebration",
    "renovation",
]

CENTURY_RANGE = list(range(11, 18))

TEMPLATE = "The {noun} lasted from the year {century}{YY:02d} to the year {century}"


def default_data_dir() -> str:
    from datasets.NLP.artifacts import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "GreaterThan")


def greater_than_json_path(data_dir: str | None = None) -> Path:
    return Path(data_dir or default_data_dir()) / "greater_than_prompts.json"


def _build_two_digit_token_map(model: Any) -> dict[int, int]:
    """Map two-digit value 0..99 -> single token id (bare ``YY``, no leading space)."""
    mapping: dict[int, int] = {}
    for yy in range(100):
        toks = model.to_tokens(f"{yy:02d}", prepend_bos=False)
        if toks.shape[1] == 1:
            mapping[yy] = int(toks[0, 0].item())
    return mapping


def generate_greater_than_prompts(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    while len(items) < n:
        noun = rng.choice(NOUNS)
        century = rng.choice(CENTURY_RANGE)
        yy = rng.randint(2, 97)
        key = (noun, century, yy)
        if key in seen:
            continue
        seen.add(key)
        prompt = TEMPLATE.format(noun=noun, century=century, YY=yy)
        items.append(
            {
                "prompt": prompt,
                "century": century,
                "YY": yy,
                "greater_tokens": list(range(yy + 1, 100)),
                "less_equal_tokens": list(range(0, yy + 1)),
            }
        )
    return items


def attach_greater_than_token_ids(
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
                **ex,
                "greater_tok_ids": greater_ids,
                "less_equal_tok_ids": leq_ids,
            }
        )
    return kept


def filter_greater_than_with_gpt2(
    items: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    """Keep rows where greedy last-token prediction lies in ``greater_tok_ids``."""
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


def save_greater_than_prompts(
    items: list[dict[str, Any]],
    data_dir: str | None = None,
) -> Path:
    out_path = greater_than_json_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_greater_than_prompts(data_dir: str | None = None) -> list[dict[str, Any]]:
    path = greater_than_json_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"GreaterThan prompts JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
