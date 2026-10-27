"""Prepare GreaterThan artifacts for a registry model_key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from saebase.datasets.GreaterThan.GreaterThan import generate_greater_than_prompts
from saebase.datasets.GreaterThan.tokenize import (
    attach_token_ids,
    filter_greedy_in_greater,
    strategy_for_model_key,
)
from saebase.datasets.prompt_schema import assert_available, make_manifest
from saebase.datasets.prompt_store import save_prompts


def prepare_for_model(
    model_key: str,
    model: Any,
    *,
    loaded_model_name: str,
    model_name: str,
    n_generate: int = 400,
    seed: int = 0,
    min_kept: int = 100,
    data_root: str | Path | None = None,
) -> Path:
    assert_available("GreaterThan", model_key)
    strategy = strategy_for_model_key(model_key)
    raw = generate_greater_than_prompts(n=n_generate, seed=seed)
    tok = attach_token_ids(raw, model, strategy=strategy)
    kept = filter_greedy_in_greater(tok, model)
    if len(kept) < min_kept:
        raise RuntimeError(
            f"GreaterThan/{model_key}: kept {len(kept)} < min_kept={min_kept}"
        )
    manifest = make_manifest(
        behavior="GreaterThan",
        model_key=model_key,
        model_name=model_name,
        loaded_model_name=loaded_model_name,
        seed=seed,
        generated_count=len(raw),
        kept_count=len(kept),
        filter_policy={
            "name": "greedy_in_greater_set",
            "tokenize_strategy": strategy,
        },
        source="datasets.GreaterThan.prepare",
        extra={"tokenized_kept": len(tok)},
    )
    return save_prompts(kept, manifest, data_root=data_root)
