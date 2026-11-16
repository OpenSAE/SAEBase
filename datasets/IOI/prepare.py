"""Prepare IOI artifacts for a registry model_key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from saebase.datasets.IOI.IOI import generate_ioi_prompts
from saebase.datasets.IOI.tokenize import attach_token_ids, filter_with_model
from saebase.datasets.artifacts import assert_available, make_manifest
from saebase.datasets.artifacts import save_prompts


def prepare_for_model(
    model_key: str,
    model: Any,
    *,
    loaded_model_name: str,
    model_name: str,
    n_generate: int = 400,
    seed: int = 0,
    min_kept: int = 100,
    stop_after: int | None = None,
    data_root: str | Path | None = None,
) -> Path:
    assert_available("IOI", model_key)
    raw = generate_ioi_prompts(n=n_generate, seed=seed)
    tok = attach_token_ids(raw, model)
    kept, policy = filter_with_model(tok, model, min_kept=min_kept, max_kept=stop_after)
    if len(kept) < min_kept:
        raise RuntimeError(f"IOI/{model_key}: kept {len(kept)} < min_kept={min_kept}")
    manifest = make_manifest(
        behavior="IOI",
        model_key=model_key,
        model_name=model_name,
        loaded_model_name=loaded_model_name,
        seed=seed,
        generated_count=len(raw),
        kept_count=len(kept),
        filter_policy={
            "name": policy,
            "require_single_token_targets": True,
            "stop_after_eligible": stop_after,
        },
        source="datasets.IOI.prepare",
        extra={"single_token_kept": len(tok)},
    )
    return save_prompts(kept, manifest, data_root=data_root)


def select_samples(cfg, *, model_key, model, loaded_model_name, seed=42):
    """Retain exactly the requested number after the existing model filter."""
    import json
    import random
    import tempfile
    from saebase.datasets.selection import SampleSelection

    if model is None:
        raise ValueError("IOI preparation requires the selected real model")
    count = int(cfg.get("sample_count", 500))
    maximum = int(cfg.get("max_candidates", 100000))
    candidates = min(maximum, max(400, count * 4))
    while True:
        with tempfile.TemporaryDirectory(prefix="saebase-ioi-") as stage:
            try:
                path = prepare_for_model(
                    model_key,
                    model,
                    loaded_model_name=loaded_model_name,
                    model_name=loaded_model_name,
                    n_generate=candidates,
                    seed=seed,
                    min_kept=count,
                    stop_after=count,
                    data_root=stage,
                )
            except RuntimeError as exc:
                if "kept" not in str(exc) or candidates >= maximum:
                    raise
                candidates = min(maximum, candidates * 2)
                continue
            rows = [
                json.loads(line)
                for line in (path / "prompts.jsonl").read_text().splitlines()
                if line.strip()
            ]
            manifest = json.loads((path / "manifest.json").read_text())
        chosen = random.Random(seed).sample(rows, count)
        for index, row in enumerate(chosen):
            row.setdefault("sample_id", f"IOI:{model_key}:{index}")
        return SampleSelection(
            chosen,
            {
                "source": "generated_argmax_correct",
                "model_key": model_key,
                "loaded_model_name": loaded_model_name,
                "generated_count": candidates,
                "eligible_count": len(rows),
                "preparation_manifest": manifest,
            },
        )
