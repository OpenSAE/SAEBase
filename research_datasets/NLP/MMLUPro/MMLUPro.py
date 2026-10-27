"""Load and render TIGER-Lab MMLU-Pro prompts for SAE last-token probes."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

CHOICES = tuple("ABCDEFGHIJKLMNOP")
HF_DATASET_NAME = "TIGER-Lab/MMLU-Pro"
CATEGORIES: frozenset[str] = frozenset(
    {
        "biology",
        "business",
        "chemistry",
        "computer science",
        "economics",
        "engineering",
        "health",
        "history",
        "law",
        "math",
        "other",
        "philosophy",
        "physics",
        "psychology",
    }
)
CATEGORY_ALIASES: dict[str, str] = {
    "computer_science": "computer science",
    "cs": "computer science",
    "all": "all",
}
INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {$}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n"
)

_DEFAULT_SOURCE_CANDIDATES = (
    Path("/workspace/data/MMLU-Pro"),
    Path(__file__).resolve().parents[3] / "data" / "MMLU-Pro",
)


def default_data_dir() -> str:
    from research_datasets.NLP.artifacts import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "MMLUPro")


def default_source_dir() -> Path | None:
    for candidate in _DEFAULT_SOURCE_CANDIDATES:
        if _source_looks_valid(candidate):
            return candidate
    return None


def _source_looks_valid(root: Path) -> bool:
    return (
        (root / "test.jsonl").is_file()
        or (root / "validation.jsonl").is_file()
        or (root / "hf_dataset").is_dir()
    )


def parse_categories(value: str | Sequence[str] | None) -> frozenset[str] | None:
    """Return canonical category set, or None for all subjects."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw = [str(part).strip() for part in value if str(part).strip()]
    if not raw or any(part.lower() == "all" for part in raw):
        return None
    out: set[str] = set()
    for part in raw:
        key = CATEGORY_ALIASES.get(part, CATEGORY_ALIASES.get(part.lower(), part))
        if key not in CATEGORIES:
            raise ValueError(f"Unknown MMLU-Pro category {part!r}. Valid: {sorted(CATEGORIES)}")
        out.add(key)
    return frozenset(out)


def drop_na_and_remap(row: dict[str, Any], *, drop_na: bool = True) -> dict[str, Any] | None:
    """Drop N/A options and remap the answer letter to the remaining list."""
    options = list(row.get("options") or [])
    try:
        idx = int(row["answer_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= idx < len(options)):
        return None
    correct_text = options[idx]
    if drop_na:
        options = [opt for opt in options if opt != "N/A"]
    if correct_text == "N/A" or correct_text not in options:
        return None
    new_idx = options.index(correct_text)
    if new_idx >= len(CHOICES):
        return None
    return {
        "question_id": row.get("question_id"),
        "question": str(row.get("question") or ""),
        "options": options,
        "answer": CHOICES[new_idx],
        "answer_index": new_idx,
        "cot_content": str(row.get("cot_content") or ""),
        "category": str(row.get("category") or ""),
        "src": str(row.get("src") or ""),
    }


def format_example(
    example: dict[str, Any],
    *,
    including_answer: bool,
    scoring: str,
) -> str:
    parts = ["Question:\n", str(example["question"]), "\nOptions:\n"]
    for i, opt in enumerate(example["options"]):
        parts.append(f"{CHOICES[i]}. {opt}\n")
    if including_answer:
        if scoring == "cot":
            cot = str(example.get("cot_content") or "").replace(
                "A: Let's think step by step.",
                "Answer: Let's think step by step.",
            )
            if cot.strip():
                parts.append(cot)
                if not cot.endswith("\n"):
                    parts.append("\n")
                parts.append("\n")
            else:
                parts.append(f"Answer: {example['answer']}\n\n")
        else:
            parts.append(f"Answer: {example['answer']}\n\n")
    elif scoring == "cot":
        parts.append("Answer: Let's think step by step.")
    else:
        parts.append("Answer:")
    return "".join(parts)


def render_prompt(
    example: dict[str, Any],
    *,
    fewshot: Sequence[dict[str, Any]] = (),
    scoring: str = "direct",
) -> str:
    scoring = str(scoring).lower()
    if scoring not in {"direct", "cot"}:
        raise ValueError(f"scoring must be 'direct' or 'cot', got {scoring!r}")
    chunks: list[str] = []
    if fewshot:
        header = INITIAL_PROMPT.replace("{$}", str(example.get("category") or "this subject"))
        chunks.append(header)
        if not header.endswith("\n"):
            chunks.append("\n")
        for shot in fewshot:
            chunks.append(format_example(shot, including_answer=True, scoring=scoring))
    chunks.append(format_example(example, including_answer=False, scoring=scoring))
    return "".join(chunks)


def _select_fewshot(
    validation: Sequence[dict[str, Any]],
    category: str,
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    same = [row for row in validation if row.get("category") == category]
    return list(same[:k])


def _balance_sample(
    rows: list[dict[str, Any]],
    max_items: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("category") or "other")].append(row)
    cats = sorted(by_cat)
    if not cats:
        return []
    per = max(1, max_items // len(cats))
    picked: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for cat in cats:
        items = list(by_cat[cat])
        rng.shuffle(items)
        picked.extend(items[:per])
        leftovers.extend(items[per:])
    if len(picked) < max_items:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: max_items - len(picked)])
    rng.shuffle(picked)
    return picked[:max_items]


def subsample_rows(
    rows: Sequence[dict[str, Any]],
    *,
    categories: str | Sequence[str] | None = "all",
    max_items: int | None = None,
    seed: int = 0,
    balance_per_category: bool = False,
) -> list[dict[str, Any]]:
    selected = list(rows)
    cats = parse_categories(categories)
    if cats is not None:
        selected = [row for row in selected if row.get("category") in cats]
    if max_items is None or max_items >= len(selected):
        return selected
    rng = random.Random(int(seed))
    if balance_per_category:
        return _balance_sample(selected, int(max_items), rng)
    order = list(range(len(selected)))
    rng.shuffle(order)
    return [selected[i] for i in order[: int(max_items)]]


def attach_distractors(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    out: list[dict[str, Any]] = []
    for row in rows:
        n_opt = len(row["options"])
        idx = int(row["answer_index"])
        wrong = [CHOICES[i] for i in range(n_opt) if i != idx]
        if not wrong:
            continue
        item = dict(row)
        item["distractor"] = rng.choice(wrong)
        out.append(item)
    return out


def _rows_from_hf_split(ds: Any, split: str) -> list[dict[str, Any]]:
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        if split in keys:
            ds = ds[split]
        elif keys:
            ds = ds[keys[0]]
    return [dict(ex) for ex in ds]


def load_raw_mmlu_pro(
    split: str = "test",
    *,
    source_path: str | Path | None = None,
    hf_name: str = HF_DATASET_NAME,
) -> list[dict[str, Any]]:
    split = str(split)
    if split not in {"test", "validation"}:
        raise ValueError(f"split must be 'test' or 'validation', got {split!r}")

    root: Path | None
    if source_path is not None:
        root = Path(source_path)
    else:
        root = default_source_dir()

    if root is not None:
        jsonl = root / f"{split}.jsonl"
        if jsonl.is_file():
            rows: list[dict[str, Any]] = []
            with jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        hf_disk = root / "hf_dataset"
        if hf_disk.is_dir():
            from models.hf_compat import import_hf_datasets

            hf_datasets = import_hf_datasets()
            ds = hf_datasets.load_from_disk(str(hf_disk))
            return _rows_from_hf_split(ds, split)

    from models.hf_compat import import_hf_datasets

    hf_datasets = import_hf_datasets()
    ds = hf_datasets.load_dataset(hf_name, split=split)
    return [dict(ex) for ex in ds]


def build_prompt_items(
    *,
    split: str = "test",
    categories: str | Sequence[str] | None = "all",
    n_shot: int = 0,
    scoring: str = "direct",
    max_items: int | None = 300,
    seed: int = 0,
    drop_na_options: bool = True,
    balance_per_category: bool = False,
    source_path: str | Path | None = None,
    hf_name: str = HF_DATASET_NAME,
    validation_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw = load_raw_mmlu_pro(split, source_path=source_path, hf_name=hf_name)
    normalized: list[dict[str, Any]] = []
    for row in raw:
        item = drop_na_and_remap(row, drop_na=drop_na_options)
        if item is not None and item["question"].strip():
            normalized.append(item)

    shots_pool: list[dict[str, Any]] = []
    if int(n_shot) > 0:
        if validation_rows is None:
            val_raw = load_raw_mmlu_pro("validation", source_path=source_path, hf_name=hf_name)
        else:
            val_raw = list(validation_rows)
        for row in val_raw:
            item = drop_na_and_remap(row, drop_na=drop_na_options)
            if item is not None:
                shots_pool.append(item)

    sampled = subsample_rows(
        normalized,
        categories=categories,
        max_items=max_items,
        seed=seed,
        balance_per_category=balance_per_category,
    )
    with_dist = attach_distractors(sampled, seed=seed)
    rendered: list[dict[str, Any]] = []
    for row in with_dist:
        fewshot = _select_fewshot(shots_pool, str(row["category"]), int(n_shot))
        rendered.append(
            {
                **row,
                "prompt": render_prompt(row, fewshot=fewshot, scoring=scoring),
            }
        )
    return rendered
