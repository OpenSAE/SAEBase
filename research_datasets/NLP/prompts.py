"""Read prepared NLP prompt artifacts from selected dataset and model.

SAEBase's canonical layout is <data_root>/<dataset>/<model>/v1/prompts.jsonl
with a matching manifest.json. Legacy JSON snapshots remain supported.
No implicit downloads, synthetic prompts, or train/test split inventions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEGACY_FILENAMES = {
    "CounterFact": ("counterfact_filtered.json", "prompts.json"),
    "GreaterThan": ("greater_than_prompts.json", "prompts.json"),
    "IOI": ("ioi_prompts.json", "prompts.json"),
    "MMLUPro": ("mmlu_pro_prompts.json", "mmlu_pro.json", "prompts.json"),
    "SimpleCopying": ("copying_prompts.json", "prompts.json"),
    "SubjectVerbAgreement": ("sva_prompts.json", "prompts.json"),
    "BBH": ("hf_cache.jsonl", "prompts.json"),
}


def prompt_file(dataset_cfg: Any, *, model_key: str | None = None) -> Path:
    name = str(dataset_cfg.name)
    if name not in LEGACY_FILENAMES:
        raise ValueError(f"Unknown NLP prompt dataset: {name}")
    root = Path(str(dataset_cfg.get("data_path", "") or "")).expanduser()
    override = str(dataset_cfg.get("prompt_file", "") or "").strip()
    if override:
        path = Path(override).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(f"{name} requires prepared prompt file {path}")
        return path
    if model_key:
        candidate = root / str(model_key) / "v1" / "prompts.jsonl"
        if candidate.is_file():
            return candidate
    else:
        matches = sorted(root.glob("*/v1/prompts.jsonl"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Specify model_key when multiple SAEBase prompt sets are present")
    for fname in LEGACY_FILENAMES[name]:
        path = root / fname
        if path.is_file():
            return path
    expected = root / str(model_key or "<model>") / "v1" / "prompts.jsonl"
    raise FileNotFoundError(f"{name} needs prepared prompt records at {expected}")


def read_prompts(dataset_cfg: Any, *, model_key: str | None = None) -> list[dict[str, Any]]:
    path = prompt_file(dataset_cfg, model_key=model_key)
    name = str(dataset_cfg.name)
    if path.suffix == ".jsonl":
        if path.name == "prompts.jsonl" and path.parent.name == "v1":
            # Real SAEBase artifact: verify the manifest identifies its source.
            from research_datasets.NLP.artifacts import load_prompts
            expected_model = path.parent.parent.name
            if model_key is not None and expected_model != model_key:
                raise ValueError("Prompt artifact model does not match selected model")
            rows = load_prompts(name, expected_model, data_root=path.parents[3],
                                require_manifest=True)
        else:
            with path.open(encoding="utf-8") as source:
                rows = [json.loads(line) for line in source if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("prompts", rows.get("rows", rows.get("samples")))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected nonempty prompt list in {path}")
    valid = []
    for index, item in enumerate(rows):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index}: expected prompt object or string")
        prompt = item.get("prompt", item.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{index}: missing prompt text")
        valid.append({**item, "prompt": prompt.strip()})
    return valid


def read_first_prompt(dataset_cfg: Any, *, model_key: str | None = None) -> str:
    return read_prompts(dataset_cfg, model_key=model_key)[0]["prompt"]
