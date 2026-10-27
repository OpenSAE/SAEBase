"""BIG-Bench Hard prompt items for discovery replication (doc 02 §28)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Prefer local mirror if present; else HF.
_LOCAL_CANDIDATES = (
    Path("/workspace/data/BBH"),
    Path(__file__).resolve().parents[3] / "data" / "BBH",
)

# A representative subset of BBH tasks (logical / abstract reasoning).
DEFAULT_SUBSETS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
    "word_sorting",
)


def _cache_bbh_from_hf(
    out_path: Path, subsets: tuple[str, ...], *, max_items: int
) -> list[dict[str, Any]]:
    """Fetch BBH in a subprocess so HuggingFace ``datasets`` is not shadowed."""
    import subprocess
    import sys

    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = r"""
import json, sys
from datasets import load_dataset
out, max_items = sys.argv[1], int(sys.argv[2])
subsets = sys.argv[3:]
rows = []
for name in subsets:
    try:
        ds = load_dataset("lukaemon/bbh", name, split="test")
    except Exception as e:
        print("skip", name, e, file=sys.stderr)
        continue
    for i, ex in enumerate(ds):
        q = str(ex.get("input") or "").strip()
        if not q:
            continue
        rows.append({
            "id": f"bbh-{name}-{i}",
            "question": q,
            "prompt": q,
            "answer": str(ex.get("target") or ""),
            "category": name,
        })
    if len(rows) >= max_items * 4:
        break
with open(out, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(rows), file=sys.stderr)
"""
    cmd = [sys.executable, "-c", script, str(out_path), str(max_items), *subsets]
    subprocess.run(
        cmd,
        check=True,
        cwd="/tmp",
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}),
            "PYTHONPATH": "",
        },
    )
    rows: list[dict[str, Any]] = []
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def default_data_dir() -> str:
    from research_datasets.NLP.artifacts import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "BBH")


def _local_jsonl(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    for path in sorted(root.glob("**/*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, list):
            rows.extend(blob)
        elif isinstance(blob, dict) and "examples" in blob:
            rows.extend(blob["examples"])
    return rows


def build_prompt_items(
    *,
    max_items: int = 80,
    seed: int = 0,
    source_path: str | None = None,
    subsets: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return discovery-ready prompt rows: id, prompt/question, answer, category."""
    subsets = subsets or DEFAULT_SUBSETS
    rows: list[dict[str, Any]] = []

    if source_path:
        root = Path(source_path)
        if root.is_dir():
            rows = _local_jsonl(root)

    if not rows:
        for cand in _LOCAL_CANDIDATES:
            if cand.is_dir():
                rows = _local_jsonl(cand)
                if rows:
                    break

    if not rows:
        cache = Path(default_data_dir()) / "hf_cache.jsonl"
        if cache.is_file():
            rows = [
                json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()
            ]
        if not rows:
            try:
                rows = _cache_bbh_from_hf(cache, subsets, max_items=int(max_items))
            except Exception as exc:
                raise RuntimeError("BBH requires huggingface datasets or local JSON/JSONL") from exc

    # Normalize local rows.
    norm: list[dict[str, Any]] = []
    for i, ex in enumerate(rows):
        q = str(ex.get("prompt") or ex.get("question") or ex.get("input") or "").strip()
        if not q:
            continue
        norm.append(
            {
                "id": ex.get("id", f"bbh-{i}"),
                "question": q,
                "prompt": q,
                "answer": ex.get("answer") or ex.get("target") or "",
                "category": ex.get("category") or ex.get("subset") or "bbh",
            }
        )

    rng = random.Random(int(seed))
    rng.shuffle(norm)
    return norm[: max(int(max_items), 0)]
