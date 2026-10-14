"""Subject-verb agreement prompt generation, token-id attachment, and JSON persistence."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

NOUN_PAIRS = [
    ("key", "keys"),
    ("door", "doors"),
    ("book", "books"),
    ("paper", "papers"),
    ("box", "boxes"),
    ("file", "files"),
    ("letter", "letters"),
    ("plate", "plates"),
    ("plant", "plants"),
    ("painting", "paintings"),
    ("photo", "photos"),
    ("chair", "chairs"),
    ("table", "tables"),
    ("window", "windows"),
    ("ticket", "tickets"),
]

ATTRACTOR_PAIRS = [
    ("cabinet", "cabinets"),
    ("drawer", "drawers"),
    ("room", "rooms"),
    ("office", "offices"),
    ("shelf", "shelves"),
    ("house", "houses"),
    ("garage", "garages"),
    ("library", "libraries"),
]

PREPS = ["to", "in", "on", "near", "from", "beside"]

TEMPLATE = "The {subj} {prep} the {attr}"


def default_data_dir() -> str:
    from saebase.data.artifact_layout import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "SubjectVerbAgreement")


def sva_json_path(data_dir: str | None = None) -> Path:
    return Path(data_dir or default_data_dir()) / "sva_prompts.json"


def generate_sva_prompts(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(items) < n:
        singular, plural = rng.choice(NOUN_PAIRS)
        subj_is_plural = rng.random() < 0.5
        subj = plural if subj_is_plural else singular

        attr_sing, attr_plur = rng.choice(ATTRACTOR_PAIRS)
        attr = attr_sing if subj_is_plural else attr_plur
        prep = rng.choice(PREPS)

        prompt = TEMPLATE.format(subj=subj, prep=prep, attr=attr)
        if prompt in seen:
            continue
        seen.add(prompt)

        correct_verb = "are" if subj_is_plural else "is"
        distractor_verb = "is" if subj_is_plural else "are"
        items.append(
            {
                "prompt": prompt,
                "subj_is_plural": subj_is_plural,
                "correct_verb": correct_verb,
                "distractor_verb": distractor_verb,
            }
        )
    return items


def attach_sva_token_ids(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ex in items:
        t_c = model.to_tokens(" " + ex["correct_verb"], prepend_bos=False)
        t_d = model.to_tokens(" " + ex["distractor_verb"], prepend_bos=False)
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


def filter_sva_with_gpt2(items: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    """Keep rows where greedy last-token prediction equals ``tok_correct``."""
    import torch  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    for ex in items:
        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        pred = int(logits.argmax().item())
        if pred == int(ex["tok_correct"]):
            kept.append(ex)
    return kept


def save_sva_prompts(items: list[dict[str, Any]], data_dir: str | None = None) -> Path:
    out_path = sva_json_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_sva_prompts(data_dir: str | None = None) -> list[dict[str, Any]]:
    path = sva_json_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"SVA prompts JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
