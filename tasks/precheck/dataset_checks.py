"""Source-owned frozen Dataset artifact Precheck, independent of any QA500 study.

This tests local, previously prepared *real* QA records and their provenance.
It does not download a dataset or treat a local file as proof of live Hub access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from tasks.precheck.report import blocked, error_result

_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_HEX40 = re.compile(r"[a-f0-9]{40}\Z")
_HEX64 = re.compile(r"[a-f0-9]{64}\Z")


def _digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _artifact(root: Path, filename: str) -> Path:
    path = root / filename
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Invalid or missing frozen cohort artifact: {filename}")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"Frozen cohort artifact exceeds bounded Precheck read: {filename}")
    return path


def check_frozen_dataset(
    *,
    dataset_id: str,
    inputs_root: Path,
    expected_count: int,
    expected_revision: str | None = None,
    renderer: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Verify source provenance, question identity and exact prompt hashes.

    The production renderer is the existing QA-record contract. Tests may
    inject a renderer to validate the algorithm with synthetic fixtures; these
    tests are not evidence that a real upstream dataset has been downloaded.
    """
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        return error_result(ValueError("Unsafe dataset identifier"), stage="dataset", attempted=[])
    if type(expected_count) is not int or not 1 <= expected_count <= 100_000:
        return error_result(ValueError("Invalid dataset sample budget"), stage="dataset", attempted=[])
    root = Path(inputs_root) / dataset_id
    if not root.is_dir() or root.is_symlink():
        return blocked(
            stage=f"dataset:{dataset_id}",
            code="BLOCKED_ASSET",
            reason="Frozen dataset directory missing or unsafe",
            attempted=["resolve frozen dataset path"],
            repair="Prepare the real pinned dataset through its authorized Dataset owner.",
        )
    if renderer is None:
        try:
            from saebase.data.qa_records import render_prompt

            renderer = render_prompt
        except ImportError as exc:
            return blocked(
                stage=f"dataset:{dataset_id}",
                code="BLOCKED_DEPENDENCY",
                reason=f"QA record contract is unavailable: {type(exc).__name__}",
                attempted=["resolve source-owned QA record renderer"],
                repair="Integrate the approved WP22 shared Dataset contract; do not invent a second renderer.",
            )
    try:
        manifest = json.loads(_artifact(root, "manifest.json").read_text(encoding="utf-8"))
        path = _artifact(root, "questions.jsonl")
        raw = path.read_bytes()
        if not isinstance(manifest, dict):
            raise ValueError("Frozen cohort manifest is not an object")
        if manifest.get("dataset_id") != dataset_id or manifest.get("n") != expected_count:
            raise ValueError("Cohort dataset identity or sample count mismatch")
        revision = manifest.get("source_revision")
        if not isinstance(revision, str) or not _HEX40.fullmatch(revision):
            raise ValueError("Source revision must be an exact 40-hex commit")
        if expected_revision is not None and revision != expected_revision:
            raise ValueError("Frozen dataset revision differs from approved source")
        declared = manifest.get("sha256")
        if not isinstance(declared, str) or not _HEX64.fullmatch(declared):
            raise ValueError("Manifest must declare the byte SHA256")
        if hashlib.sha256(raw).hexdigest() != declared:
            raise ValueError("Frozen cohort bytes differ from manifest")
        lines = raw.decode("utf-8").splitlines()
        if len(lines) != expected_count or any(not line for line in lines):
            raise ValueError("Frozen cohort has missing, extra or empty records")
        records = [json.loads(line) for line in lines]
        ids = []
        prompts = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Frozen QA record is not an object")
            if record.get("dataset_id") != dataset_id or record.get("source_revision") != revision:
                raise ValueError("Question does not match dataset and source revision")
            if record.get("source_repo") != manifest.get("source_repo"):
                raise ValueError("Question source repository differs from manifest")
            if (
                manifest.get("source_config") is not None
                and record.get("source_config") != manifest.get("source_config")
            ):
                raise ValueError("Question source configuration differs from manifest")
            if record.get("source_split") != manifest.get("source_split"):
                raise ValueError("Question split differs from frozen source")
            for name in ("sample_id", "prompt_sha256", "question", "answer"):
                if not isinstance(record.get(name), str) or not record[name].strip():
                    raise ValueError(f"QA record missing {name}")
            if record.get("answer_type") == "choice":
                choices = record.get("choices")
                if not isinstance(choices, list) or record["answer"] not in {
                    x.get("label") for x in choices if isinstance(x, dict)
                }:
                    raise ValueError("Choice answer does not resolve to an option")
            prompt = renderer(record)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("QA prompt renderer returned empty input")
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record["prompt_sha256"]:
                raise ValueError("QA prompt digest mismatch")
            ids.append(record["sample_id"])
            prompts.append(record["prompt_sha256"])
        if len(set(ids)) != expected_count or len(set(prompts)) != expected_count:
            raise ValueError("Frozen QA sample or prompt identity duplicated")
        if _digest(ids) != manifest.get("sample_ids_sha256"):
            raise ValueError("Frozen QA sample ID list differs from manifest")
        return {
            "status": "PASS",
            "stage": f"dataset:{dataset_id}",
            "scope": "local frozen real QA record integrity only, not upstream or scientific acceptance",
            "dataset_id": dataset_id,
            "expected": expected_count,
            "observed": len(records),
            "source_repo": manifest.get("source_repo"),
            "source_revision": revision,
            "cohort_sha256": declared,
            "sample_ids_sha256": manifest["sample_ids_sha256"],
            "source_lock_status": manifest.get("lock_status", "UNKNOWN"),
            "scientific_acceptance": "NOT_COVERED",
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        return error_result(
            exc,
            stage=f"dataset:{dataset_id}",
            attempted=["read local frozen cohort", "validate actual bytes, identities and prompts"],
        )
