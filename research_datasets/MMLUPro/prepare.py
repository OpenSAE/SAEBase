"""Prepare MMLU-Pro artifacts for a registry model_key."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from research_datasets.NLP.MMLUPro.MMLUPro import HF_DATASET_NAME, build_prompt_items
from research_datasets.NLP.MMLUPro.tokenize import attach_token_ids
from research_datasets.NLP.artifacts import assert_available, make_manifest
from research_datasets.NLP.artifacts import save_prompts


def prepare_for_model(
    model_key: str,
    model: Any,
    *,
    loaded_model_name: str,
    model_name: str,
    n_generate: int = 300,
    seed: int = 0,
    min_kept: int = 20,
    data_root: str | Path | None = None,
    split: str = "test",
    categories: str | Sequence[str] | None = "all",
    n_shot: int = 0,
    scoring: str = "direct",
    max_prompt_tokens: int | None = 2048,
    drop_na_options: bool = True,
    balance_per_category: bool = False,
    source_path: str | Path | None = None,
    hf_name: str = HF_DATASET_NAME,
    letter_prefix: str = " ",
) -> Path:
    assert_available("MMLUPro", model_key)
    raw = build_prompt_items(
        split=split,
        categories=categories,
        n_shot=n_shot,
        scoring=scoring,
        max_items=n_generate,
        seed=seed,
        drop_na_options=drop_na_options,
        balance_per_category=balance_per_category,
        source_path=source_path,
        hf_name=hf_name,
    )
    kept = attach_token_ids(
        raw,
        model,
        letter_prefix=letter_prefix,
        max_prompt_tokens=max_prompt_tokens,
    )
    if len(kept) < min_kept:
        raise RuntimeError(f"MMLUPro/{model_key}: kept {len(kept)} < min_kept={min_kept}")
    cats = categories if categories is not None else "all"
    if not isinstance(cats, str):
        cats = ",".join(str(c) for c in cats)
    manifest = make_manifest(
        behavior="MMLUPro",
        model_key=model_key,
        model_name=model_name,
        loaded_model_name=loaded_model_name,
        seed=seed,
        generated_count=len(raw),
        kept_count=len(kept),
        filter_policy={
            "name": "single_token_letter",
            "require_single_token_targets": True,
            "max_prompt_tokens": max_prompt_tokens,
            "drop_na_options": drop_na_options,
        },
        source="research_datasets.NLP.MMLUPro.prepare",
        extra={
            "split": split,
            "categories": cats,
            "n_shot": int(n_shot),
            "scoring": scoring,
            "max_items": int(n_generate),
            "balance_per_category": bool(balance_per_category),
            "hf_name": hf_name,
        },
    )
    return save_prompts(kept, manifest, data_root=data_root)
