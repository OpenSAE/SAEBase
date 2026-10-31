"""Frozen cohort for the QA500 experiment.

Dataset schemas live in source-owned Dataset adapters. The experiment selects
its dataset instances through the canonical SAEBase Dataset Registry.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
from typing import Any
from saebase.data.qa_records import DatasetSpec, digest, render_prompt, make_record
from saebase.data.qa_source import iter_source as _load_source
from saebase.registry import get_qa_dataset_adapter
from experiments.qa500.config import get_protocol, get_dataset_ids
from experiments.qa500.assets import source_revision

PROTOCOL = get_protocol()
PROTOCOL_ID = PROTOCOL["protocol_id"]
SEED = int(PROTOCOL["seed"])
N_PER_DATASET = int(PROTOCOL["samples_per_dataset"])
DATASETS: dict[str, DatasetSpec] = {
    dataset_id: get_qa_dataset_adapter(dataset_id).SPEC for dataset_id in get_dataset_ids()
}


def normalize(
    spec: DatasetSpec,
    original: dict[str, Any],
    *,
    source_config: str | None,
    source_index: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Invoke the relevant Dataset's own schema parser."""
    adapter = get_qa_dataset_adapter(spec.key)
    parsed = adapter.parse(dict(original), source_config=source_config, seed=seed)
    return make_record(spec, original, parsed, source_config, source_index)


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
    revision, lock_status = source_revision(spec.repo, spec.key)
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
        "lock_status": lock_status,
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
