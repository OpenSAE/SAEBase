"""Real locally prepared Dataset prompt-artifact integrity checks.

A source without prepared real records is BLOCKED, never synthetic PASS.
The Dataset owner performs loading and manifest validation; this module checks
actual bytes, record IDs and prompt integrity without a second Dataset API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tasks.precheck.report import blocked, error_result


def check_dataset_artifact(
    *, dataset_key: str, model_key: str, data_root: Path,
    max_records: int = 500,
) -> dict[str, Any]:
    """Validate actual model-specific Dataset records with bounded reading."""
    if not 1 <= max_records <= 10000:
        raise ValueError("max_records must be bounded")
    try:
        from omegaconf import OmegaConf
        from saebase.registry import get_dataset_plugin

        get_dataset_plugin(dataset_key)
        from saebase.datasets.prompts import prompt_file, read_prompts

        dataset_cfg = OmegaConf.create({
            "name": dataset_key,
            "data_path": str(Path(data_root) / dataset_key),
            "prompt_file": "",
        })
        path = prompt_file(dataset_cfg, model_key=model_key)
        if path.is_symlink() or not path.resolve().is_relative_to(Path(data_root).resolve()):
            raise ValueError("Unsafe Dataset artifact path")
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Dataset artifact exceeds Precheck byte bound")
        rows = read_prompts(dataset_cfg, model_key=model_key)
        if len(rows) > 100000:
            raise ValueError("Dataset record count exceeds safety bound")
        ids, prompts = [], []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("Dataset record must be a mapping")
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Dataset prompt missing")
            sample_id = str(row.get("sample_id") or hashlib.sha256(
                (json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)).encode()
            ).hexdigest())
            ids.append(sample_id)
            prompts.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
            raise ValueError("Duplicate Dataset sample or prompt")
        raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "status": "PASS",
            "scope": "local real Dataset prompt artifact only; not upstream or scientific acceptance",
            "dataset_id": dataset_key,
            "model_key": model_key,
            "observed": len(rows),
            "max_records_for_worker": max_records,
            "path": str(path),
            "artifact_sha256": raw_sha,
            "sample_ids_sha256": hashlib.sha256(
                json.dumps(ids, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "scientific_acceptance": "NOT_COVERED",
        }
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as exc:
        return blocked(
            stage=f"dataset:{dataset_key}:{model_key}",
            code="BLOCKED_ASSET",
            reason=f"Real Dataset source or dependency unavailable: {type(exc).__name__}",
            attempted=["resolve source-owned Dataset artifact"],
            repair="Prepare the actual approved Dataset asset before Precheck.",
        )
    except Exception as exc:
        return error_result(
            exc, stage=f"dataset:{dataset_key}:{model_key}",
            attempted=["read real Dataset records and verify identities"],
        )
