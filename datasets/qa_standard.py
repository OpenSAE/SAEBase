"""Frozen, model-independent Hugging Face QA cohort for SAEBase.

This module intentionally lives in the source-owned datasets/ tree.  Use it through
saebase.datasets.qa_standard so the installed wheel and source checkout share imports.
No implicit synthetic data, fallback dataset, answer leakage or silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from string import ascii_uppercase
from typing import Any, Iterable


PROTOCOL_ID = "qa_standard_500_v1"
SEED = 20261031
N_PER_DATASET = 500


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo: str
    config: str | None
    split: str
    kind: str
    restricted: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "mmlu_pro": DatasetSpec("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", "mmlu"),
    "arc_challenge": DatasetSpec(
        "arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "arc"
    ),
    "commonsense_qa": DatasetSpec(
        "commonsense_qa", "tau/commonsense_qa", None, "validation", "commonsense"
    ),
    "boolq": DatasetSpec("boolq", "google/boolq", None, "validation", "boolq"),
    "bbh": DatasetSpec("bbh", "lukaemon/bbh", None, "test", "bbh"),
    "gsm8k": DatasetSpec("gsm8k", "openai/gsm8k", "main", "test", "gsm8k"),
    "math500": DatasetSpec("math500", "HuggingFaceH4/MATH-500", None, "test", "math500"),
    "medqa": DatasetSpec("medqa", "GBaker/MedQA-USMLE-4-options", None, "test", "medqa"),
    "hotpot_qa": DatasetSpec(
        "hotpot_qa", "hotpotqa/hotpot_qa", "distractor", "validation", "hotpot"
    ),
    "gpqa_extended": DatasetSpec(
        "gpqa_extended", "Idavidrein/gpqa", "gpqa_extended", "train", "gpqa", True
    ),
}


def digest(value: Any) -> str:
    """Canonical SHA256 of JSON data, independent of dictionary insertion order."""
    blob = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _string(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _choices(pairs: Iterable[tuple[str, Any]], gold_key: str) -> tuple[list[dict[str, str]], str]:
    """Remap real upstream choice labels to A/B/... without changing the correct answer."""
    options: list[dict[str, str]] = []
    canonical_gold = ""
    for index, (label, text) in enumerate(pairs):
        if index >= 16:
            raise ValueError("Too many multiple choice options")
        content = _string(text)
        if not content or content == "N/A":
            continue
        new_label = ascii_uppercase[len(options)]
        options.append({"label": new_label, "text": content})
        if _string(label).upper() == _string(gold_key).upper():
            canonical_gold = new_label
    if len(options) < 2 or canonical_gold not in {x["label"] for x in options}:
        raise ValueError("Invalid choice options or gold answer")
    return options, canonical_gold


def normalize(
    spec: DatasetSpec,
    original: dict[str, Any],
    *,
    source_config: str | None,
    source_index: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Normalize a *real* source item; fail rather than substituting question/answer."""
    data = dict(original)
    kind = spec.kind
    question = _string(
        data.get("question") or data.get("problem") or data.get("input") or data.get("Question")
    )
    context = ""
    options: list[dict[str, str]] = []
    answer_type = "exact_text"
    answer = ""
    category = _string(data.get("category") or data.get("type") or source_config or "all")
    if kind == "mmlu":
        raw = list(data.get("options") or [])
        key = data.get("answer_index")
        if key is None:
            key = ascii_uppercase.index(_string(data.get("answer")).upper())
        options, answer = _choices(
            [(ascii_uppercase[i], item) for i, item in enumerate(raw)], ascii_uppercase[int(key)]
        )
        answer_type = "choice"
    elif kind == "arc":
        choices = data.get("choices") or {}
        pairs = list(zip(choices.get("label") or [], choices.get("text") or []))
        options, answer = _choices(pairs, _string(data.get("answerKey")))
        answer_type = "choice"
    elif kind == "commonsense":
        choices = data.get("choices") or {}
        options, answer = _choices(
            zip(choices.get("label") or [], choices.get("text") or []),
            _string(data.get("answerKey")),
        )
        answer_type = "choice"
    elif kind == "boolq":
        context = _string(data.get("passage"))
        if type(data.get("answer")) is not bool:
            raise ValueError("BoolQ answer must be boolean")
        answer = "Yes" if data["answer"] else "No"
        answer_type = "boolean"
    elif kind == "bbh":
        answer = _string(data.get("target"))
    elif kind == "gsm8k":
        solution = _string(data.get("answer"))
        if "####" not in solution:
            raise ValueError("GSM8K final answer delimiter missing")
        answer = solution.rsplit("####", 1)[1].strip().replace(",", "")
        answer_type = "number"
    elif kind == "math500":
        answer = _string(data.get("answer"))
        answer_type = "math"
    elif kind == "medqa":
        opts = data.get("options") or {}
        if not isinstance(opts, dict):
            raise ValueError("MedQA options must be a label-to-text object")
        options, answer = _choices(sorted(opts.items()), _string(data.get("answer_idx")))
        answer_type = "choice"
    elif kind == "hotpot":
        ctx = data.get("context") or {}
        titles, sentences = ctx.get("title") or [], ctx.get("sentences") or []
        if len(titles) != len(sentences):
            raise ValueError("Hotpot context titles and sentences are misaligned")
        context = "\n".join(
            f"{_string(title)}: {' '.join(map(str, block))}"
            for title, block in zip(titles, sentences)
        )
        answer = _string(data.get("answer"))
    elif kind == "gpqa":
        question = _string(data.get("Question"))
        correct = _string(data.get("Correct Answer"))
        others = [_string(data.get(f"Incorrect Answer {i}")) for i in (1, 2, 3)]
        if not correct or any(not x for x in others):
            raise ValueError("GPQA answer options missing")
        values = [correct, *others]
        item_seed = int(digest([seed, question])[:12], 16)
        random.Random(item_seed).shuffle(values)
        options, answer = _choices(
            [(str(i), value) for i, value in enumerate(values)],
            str(values.index(correct)),
        )
        answer_type = "choice"
        category = _string(data.get("High-level domain") or data.get("Subdomain") or "science")
    else:
        raise KeyError(f"Unsupported QA normalization kind: {kind}")
    if not question or not answer:
        raise ValueError("Question or reference answer is empty")
    source_id = _string(
        data.get("question_id")
        or data.get("id")
        or data.get("_id")
        or data.get("Record ID")
        or source_index
    )
    # Include the upstream config: source indices are only unique within a config.
    sample_id = digest([spec.key, source_config, spec.split, source_id, question])[:24]
    record = {
        "sample_id": sample_id,
        "dataset_id": spec.key,
        "source_repo": spec.repo,
        "source_config": source_config,
        "source_split": spec.split,
        "source_id": source_id,
        "question": question,
        "context": context,
        "choices": options,
        "answer": answer,
        "answer_type": answer_type,
        "category": category,
        "prompt_group_id": sample_id,
        "restricted_source": spec.restricted,
    }
    prompt = render_prompt(record)
    if len(prompt) > 12000:
        raise ValueError("Prompt exceeds frozen 12,000-character ceiling")
    record["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return record


def render_prompt(record: dict[str, Any]) -> str:
    """A frozen zero-shot text template; answers are never included in the input."""
    kind = record["answer_type"]
    context = _string(record.get("context"))
    header = (
        "Select the correct answer. Respond with only the option letter.\n"
        if kind == "choice"
        else "Answer the question. Give only the final answer.\n"
    )
    parts = [header]
    if context:
        parts.append(f"Context:\n{context}\n")
    parts.append(f"Question:\n{record['question']}\n")
    if kind == "choice":
        parts.extend(f"{c['label']}. {c['text']}\n" for c in record["choices"])
        parts.append("Answer:")
    elif kind == "boolean":
        parts.append("Answer (Yes or No):")
    else:
        parts.append("Final answer:")
    return "".join(parts)


def _load_source(spec: DatasetSpec, revision: str) -> list[tuple[str | None, int, dict[str, Any]]]:
    from datasets import get_dataset_config_names, load_dataset

    configs = (
        sorted(get_dataset_config_names(spec.repo, revision=revision))
        if spec.kind == "bbh"
        else [spec.config]
    )
    sources: list[tuple[str | None, int, dict[str, Any]]] = []
    if spec.kind == "bbh" and len(configs) < 20:
        raise RuntimeError("BBH must expose its multiple real task configurations")
    for config in configs:
        ds = load_dataset(spec.repo, name=config, split=spec.split, revision=revision)
        for i, row in enumerate(ds):
            sources.append((config, i, dict(row)))
    return sources


def _balanced_select(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Sample round-robin over categories, with deterministic reshuffling."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["category"], []).append(record)
    rng = random.Random(seed)
    for group in groups.values():
        group.sort(key=lambda row: row["sample_id"])
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while len(selected) < n and any(groups.values()):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == n:
                    break
    if len(selected) != n:
        raise RuntimeError(f"Only {len(selected)} valid distinct records available; need {n}")
    return sorted(selected, key=lambda row: row["sample_id"])


def prepare_dataset(
    dataset_id: str,
    output_root: str | Path,
    *,
    seed: int = SEED,
    n: int = N_PER_DATASET,
    force: bool = False,
) -> dict[str, Any]:
    """Download exactly one pinned dataset and freeze 500 QA rows locally.

    This is a controlled, *explicit* network operation. On gated datasets it
    raises and preserves the reason; it never silently changes the benchmark.
    """
    from huggingface_hub import HfApi

    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_id}; choices: {sorted(DATASETS)}")
    if n != N_PER_DATASET:
        raise ValueError("The standard_500_v1 publication cohort requires exactly 500 questions")
    spec = DATASETS[dataset_id]
    root = Path(output_root) / dataset_id
    path = root / "questions.jsonl"
    manifest_path = root / "manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError("Partial previous cohort found; repair manually")
        if not force:
            return verify_dataset(dataset_id, output_root)
    revision = HfApi().dataset_info(spec.repo).sha
    if not revision or len(revision) != 40:
        raise RuntimeError("Unable to resolve exact upstream HF dataset commit")
    raw = _load_source(spec, revision)
    normal: list[dict[str, Any]] = []
    invalid = 0
    seen: set[str] = set()
    seen_prompts: set[str] = set()
    for config, index, row in raw:
        try:
            record = normalize(spec, row, source_config=config, source_index=index, seed=seed)
        except (ValueError, KeyError, TypeError, IndexError):
            invalid += 1
            continue
        if record["sample_id"] not in seen and record["prompt_sha256"] not in seen_prompts:
            seen.add(record["sample_id"])
            seen_prompts.add(record["prompt_sha256"])
            normal.append(record)
    if len(normal) < n:
        raise RuntimeError(f"{dataset_id}: only {len(normal)} usable unique rows, need {n}")
    selected = _balanced_select(normal, n, seed)
    for row in selected:
        row["source_revision"] = revision
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in selected
    ]
    payload = "\n".join(lines) + "\n"
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": dataset_id,
        "n": n,
        "seed": seed,
        "source_repo": spec.repo,
        "source_config": spec.config,
        "source_split": spec.split,
        "source_revision": revision,
        "restricted_source": spec.restricted,
        "invalid_source_rows": invalid,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "sample_ids_sha256": digest([r["sample_id"] for r in selected]),
        "format": "saebase.qa.inputs.v1",
    }
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp_manifest.replace(manifest_path)
    return verify_dataset(dataset_id, output_root)


def verify_dataset(dataset_id: str, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root) / dataset_id
    p = root / "questions.jsonl"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise RuntimeError(f"{dataset_id}: dataset manifest checksum mismatch")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(records) != N_PER_DATASET or manifest["n"] != N_PER_DATASET:
        raise RuntimeError(f"{dataset_id}: expected exactly {N_PER_DATASET} actual records")
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids) or digest(ids) != manifest["sample_ids_sha256"]:
        raise RuntimeError(f"{dataset_id}: duplicate/changed question identities")
    if len({r["prompt_sha256"] for r in records}) != N_PER_DATASET:
        raise RuntimeError(f"{dataset_id}: duplicated rendered QA prompts")
    for r in records:
        if r["dataset_id"] != dataset_id or r["source_revision"] != manifest["source_revision"]:
            raise RuntimeError("Question source identity mismatch")
        if hashlib.sha256(render_prompt(r).encode("utf-8")).hexdigest() != r["prompt_sha256"]:
            raise RuntimeError("Question prompt identity mismatch")
        if r["answer_type"] == "choice" and r["answer"] not in {c["label"] for c in r["choices"]}:
            raise RuntimeError("Choice answer does not resolve to an option")
    if manifest["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Unexpected QA protocol version")
    return manifest
