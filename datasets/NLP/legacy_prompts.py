"""Load prepared prompt artifacts for SAE evaluation."""

from __future__ import annotations

from typing import Any

from datasets.NLP.artifacts import canonicalize_behavior
from datasets.NLP.artifacts import load_manifest, load_prompts
from saebase.sae.contracts import assert_manifest_matches


def load_behavior_prompts(
    behavior: str,
    model_key: str,
    *,
    n: int | None,
    data_root: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    beh = canonicalize_behavior(behavior)
    manifest = load_manifest(beh, model_key, data_root=data_root)
    assert_manifest_matches(manifest, behavior=beh, model_key=model_key)
    rows = load_prompts(beh, model_key, data_root=data_root, require_manifest=False)
    if n is not None:
        rows = rows[: int(n)]
    if not rows:
        raise RuntimeError(f"No prompts loaded for {beh}/{model_key}")
    return manifest, rows
