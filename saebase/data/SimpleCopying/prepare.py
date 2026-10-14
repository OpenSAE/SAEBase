"""Prepare SimpleCopying artifacts for a registry model_key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from saebase.data.SimpleCopying.SimpleCopying import generate_copying_prompts
from saebase.data.SimpleCopying.tokenize import attach_token_ids, filter_logit_diff
from saebase.data.prompt_schema import assert_available, make_manifest
from saebase.data.prompt_store import save_prompts


def prepare_for_model(
    model_key: str,
    model: Any,
    *,
    loaded_model_name: str,
    model_name: str,
    n_generate: int = 300,
    seed: int = 0,
    min_kept: int = 100,
    data_root: str | Path | None = None,
) -> Path:
    assert_available("SimpleCopying", model_key)
    raw = generate_copying_prompts(n=n_generate, seed=seed)
    tok = attach_token_ids(raw, model)
    kept = filter_logit_diff(tok, model)
    if len(kept) < min_kept:
        raise RuntimeError(
            f"SimpleCopying/{model_key}: kept {len(kept)} < min_kept={min_kept}"
        )
    manifest = make_manifest(
        behavior="SimpleCopying",
        model_key=model_key,
        model_name=model_name,
        loaded_model_name=loaded_model_name,
        seed=seed,
        generated_count=len(raw),
        kept_count=len(kept),
        filter_policy={
            "name": "logit_diff",
            "require_single_token_targets": True,
        },
        source="datasets.SimpleCopying.prepare",
        extra={"single_token_kept": len(tok)},
    )
    return save_prompts(kept, manifest, data_root=data_root)
