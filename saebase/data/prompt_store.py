"""JSONL + manifest I/O for SAEBase prompt artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saebase.data.artifact_layout import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_PAIRED,
    artifact_dir,
    canonicalize_behavior,
    manifest_path,
    prompts_path,
)
from saebase.data.prompt_schema import assert_available

_ALLOWED_SCHEMA = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_PAIRED})


def save_prompts(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    data_root: str | Path | None = None,
) -> Path:
    behavior = canonicalize_behavior(str(manifest["behavior"]))
    model_key = str(manifest["model_key"])
    assert_available(behavior, model_key)
    schema_v = int(manifest.get("schema_version", -1))
    if schema_v not in _ALLOWED_SCHEMA:
        raise ValueError(
            f"manifest schema_version={manifest.get('schema_version')} "
            f"not in {sorted(_ALLOWED_SCHEMA)}"
        )
    if manifest.get("model_key") != model_key:
        raise ValueError("manifest.model_key mismatch")

    out_dir = artifact_dir(behavior, model_key, data_root=data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = prompts_path(behavior, model_key, data_root=data_root)
    man_file = manifest_path(behavior, model_key, data_root=data_root)

    with prompts_file.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    man_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Back-compat: callers historically expect the artifact directory Path.
    return out_dir


def load_manifest(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    beh = canonicalize_behavior(behavior)
    assert_available(beh, model_key)
    path = manifest_path(beh, model_key, data_root=data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt manifest not found for {beh}/{model_key}: {path}. "
            "Run scripts/sae_prepare_prompts.py first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if str(manifest.get("model_key")) != str(model_key):
        raise RuntimeError(
            f"Artifact model_key mismatch: manifest={manifest.get('model_key')!r} "
            f"requested={model_key!r}"
        )
    if canonicalize_behavior(str(manifest.get("behavior", beh))) != beh:
        raise RuntimeError(
            f"Artifact behavior mismatch: manifest={manifest.get('behavior')!r} "
            f"requested={beh!r}"
        )
    return manifest


def load_prompts(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
    require_manifest: bool = True,
) -> list[dict[str, Any]]:
    beh = canonicalize_behavior(behavior)
    if require_manifest:
        load_manifest(beh, model_key, data_root=data_root)
    path = prompts_path(beh, model_key, data_root=data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt artifact not found for {beh}/{model_key}: {path}"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
