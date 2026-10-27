"""IOI prompt generation, token-id attachment, and JSON persistence."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Name list from Wang et al. 2022 (IOI literature)
NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Zachary",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

OBJECTS = ["drink", "book", "kiss", "snack", "ring", "ball", "bone", "gift"]
PLACES = ["store", "garden", "school", "park", "office", "airport"]

TEMPLATE = "When {a} and {b} went to the {place}, {b} gave a {obj} to"


def default_data_dir() -> str:
    from saebase.datasets.artifact_layout import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "IOI")


def ioi_json_path(data_dir: str | None = None) -> Path:
    return Path(data_dir or default_data_dir()) / "ioi_prompts.json"


def generate_ioi_prompts(n: int, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    while len(items) < n:
        a, b = rng.sample(NAMES, 2)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        key = (a, b, place, obj)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "prompt": TEMPLATE.format(a=a, b=b, place=place, obj=obj),
                "io_name": a,
                "s_name": b,
            }
        )
    return items


def attach_ioi_token_ids(items: list[dict[str, str]], model: Any) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ex in items:
        t_io = model.to_tokens(" " + ex["io_name"], prepend_bos=False)
        t_s = model.to_tokens(" " + ex["s_name"], prepend_bos=False)
        if t_io.shape[1] != 1 or t_s.shape[1] != 1:
            continue
        kept.append(
            {
                **ex,
                "tok_io": int(t_io[0, 0].item()),
                "tok_s": int(t_s[0, 0].item()),
            }
        )
    return kept


def save_ioi_prompts(items: list[dict[str, Any]], data_dir: str | None = None) -> Path:
    out_path = ioi_json_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_ioi_prompts(data_dir: str | None = None) -> list[dict[str, Any]]:
    path = ioi_json_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"IOI prompts JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
