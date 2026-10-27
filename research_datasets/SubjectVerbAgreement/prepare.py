"""Prepare SubjectVerbAgreement artifacts for a registry model_key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_datasets.NLP.SubjectVerbAgreement.SubjectVerbAgreement import generate_sva_prompts
from research_datasets.NLP.SubjectVerbAgreement.tokenize import attach_token_ids, filter_with_model
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
    min_kept: int = 100,
    data_root: str | Path | None = None,
) -> Path:
    assert_available("SubjectVerbAgreement", model_key)
    raw = generate_sva_prompts(n=n_generate, seed=seed)
    tok = attach_token_ids(raw, model)
    kept, policy = filter_with_model(tok, model, min_kept=min_kept)
    if len(kept) < min_kept:
        raise RuntimeError(
            f"SubjectVerbAgreement/{model_key}: kept {len(kept)} < min_kept={min_kept}"
        )
    manifest = make_manifest(
        behavior="SubjectVerbAgreement",
        model_key=model_key,
        model_name=model_name,
        loaded_model_name=loaded_model_name,
        seed=seed,
        generated_count=len(raw),
        kept_count=len(kept),
        filter_policy={
            "name": policy,
            "require_single_token_targets": True,
        },
        source="datasets.SubjectVerbAgreement.prepare",
        extra={"single_token_kept": len(tok)},
    )
    return save_prompts(kept, manifest, data_root=data_root)
