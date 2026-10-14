"""Simple copying (A B A -> B) prompt generation and JSON persistence."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

NAMES = [
    "Mary",
    "John",
    "Sarah",
    "James",
    "Rachel",
    "David",
    "Emily",
    "Michael",
    "Laura",
    "Robert",
    "Jessica",
    "Daniel",
    "Lisa",
    "Andrew",
    "Megan",
    "Kevin",
    "Heather",
    "Brian",
    "Nicole",
    "Brandon",
    "Amanda",
    "Ryan",
    "Stephanie",
    "Jason",
    "Christine",
    "Justin",
    "Erica",
    "Aaron",
    "Crystal",
    "Eric",
    "Tiffany",
    "Adam",
    "Christina",
    "Kyle",
    "Matthew",
    "Lauren",
    "Steven",
    "Rebecca",
    "Mark",
    "Samantha",
    "Patrick",
    "Erin",
    "Sean",
    "Kelly",
]

TEMPLATE = "{a} {b} {a}"


def default_data_dir() -> str:
    from saebase.data.artifact_layout import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "SimpleCopying")


def copying_json_path(data_dir: str | None = None) -> Path:
    return Path(data_dir or default_data_dir()) / "copying_prompts.json"


def generate_copying_prompts(n: int, seed: int = 0) -> list[dict[str, str]]:
    max_prompts = len(NAMES) * (len(NAMES) - 1)
    if n < 1 or n > max_prompts:
        raise ValueError(f"n must be between 1 and {max_prompts} distinct name pairs, got {n}")
    rng = random.Random(seed)
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    while len(items) < n:
        a, b = rng.sample(NAMES, 2)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "prompt": TEMPLATE.format(a=a, b=b),
                "name_a": a,
                "name_b": b,
                "correct_name": b,
            }
        )

    for ex in items:
        candidates = [name for name in NAMES if name != ex["name_a"] and name != ex["name_b"]]
        ex["distractor_name"] = rng.choice(candidates)
    return items


def attach_copying_token_ids(items: list[dict[str, str]], model: Any) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ex in items:
        t_c = model.to_tokens(" " + ex["correct_name"], prepend_bos=False)
        t_d = model.to_tokens(" " + ex["distractor_name"], prepend_bos=False)
        if t_c.shape[1] != 1 or t_d.shape[1] != 1:
            continue
        kept.append(
            {
                **ex,
                "tok_correct": int(t_c[0, 0].item()),
                "tok_distractor": int(t_d[0, 0].item()),
            }
        )
    return kept


def filter_copying_with_gpt2(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    """Weak filter: logit(correct) > logit(distractor) at last position."""
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        if float(logits[int(ex["tok_correct"])]) > float(logits[int(ex["tok_distractor"])]):
            kept.append(ex)
    return kept


def save_copying_prompts(items: list[dict[str, Any]], data_dir: str | None = None) -> Path:
    out_path = copying_json_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_copying_prompts(data_dir: str | None = None) -> list[dict[str, Any]]:
    path = copying_json_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"SimpleCopying prompts JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
