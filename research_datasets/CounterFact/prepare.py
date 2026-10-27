"""Prepare CounterFact artifacts for a registry model_key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_datasets.NLP.CounterFact.CounterFact import load_raw_counterfact
from research_datasets.NLP.CounterFact.tokenize import filter_with_model
from research_datasets.NLP.artifacts import assert_available, make_manifest
from research_datasets.NLP.artifacts import save_prompts


def prepare_for_model(
    model_key: str,
    model: Any,
    *,
    loaded_model_name: str,
    model_name: str,
    seed: int = 0,
    min_kept: int = 100,
    max_items: int | None = 20_000,
    data_root: str | Path | None = None,
) -> Path:
    assert_available("CounterFact", model_key)
    raw = load_raw_counterfact()
    kept, policy = filter_with_model(raw, model, min_kept=min_kept, max_items=max_items)
    if len(kept) < min_kept:
        raise RuntimeError(f"CounterFact/{model_key}: kept {len(kept)} < min_kept={min_kept}")
    manifest = make_manifest(
        behavior="CounterFact",
        model_key=model_key,
        model_name=model_name,
        loaded_model_name=loaded_model_name,
        seed=seed,
        generated_count=len(raw),
        kept_count=len(kept),
        filter_policy={
            "name": policy,
            "require_single_token_targets": True,
            "max_items": max_items,
        },
        source="research_datasets.NLP.CounterFact.prepare",
    )
    return save_prompts(kept, manifest, data_root=data_root)
