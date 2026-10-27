"""CounterFact load, normalize, filter, and persist for SAEBase preparation."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

ROME_COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
HF_DATASET_NAME = "azhx/counterfact"


def default_data_dir() -> str:
    from saebase.datasets.artifact_layout import DEFAULT_DATA_ROOT

    return str(DEFAULT_DATA_ROOT / "CounterFact")


def filtered_json_path(data_dir: str | None = None) -> Path:
    root = Path(data_dir or default_data_dir())
    return root / "counterfact_filtered.json"


def _token_str(field: Any) -> str:
    if isinstance(field, dict):
        return str(field.get("str", field.get("id", "")))
    return str(field)


def normalize_counterfact_schema(raw: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map heterogeneous rows to ``prompt``, ``target_true``, ``target_false``."""
    items: list[dict[str, str]] = []
    for ex in raw:
        if "requested_rewrite" in ex:
            rr = ex["requested_rewrite"]
        elif "prompt" in ex and "subject" in ex:
            rr = ex
        else:
            continue
        subject = str(rr.get("subject", ""))
        prompt_tmpl = str(rr.get("prompt", ""))
        if "{}" not in prompt_tmpl and subject:
            prompt = prompt_tmpl.replace(subject, "{}").format(subject)
        else:
            prompt = prompt_tmpl.format(subject)
        target_true = _token_str(rr.get("target_true", ""))
        target_false = _token_str(rr.get("target_new", rr.get("target_false", "")))
        items.append(
            {
                "prompt": prompt,
                "target_true": target_true.strip(),
                "target_false": target_false.strip(),
            }
        )
    return items


def load_raw_counterfact() -> list[dict[str, str]]:
    """HF first; on failure download ROME JSON and normalize."""
    try:
        from models.hf_compat import import_hf_datasets

        hf_datasets = import_hf_datasets()
        ds = hf_datasets.load_dataset(HF_DATASET_NAME, split="train")
        raw = [dict(ex) for ex in ds]
        return normalize_counterfact_schema(raw)
    except Exception as hf_err:
        print(f"[CounterFact] HF load failed ({HF_DATASET_NAME}): {hf_err}")
        print(f"[CounterFact] Falling back to ROME URL: {ROME_COUNTERFACT_URL}")

    with urllib.request.urlopen(ROME_COUNTERFACT_URL, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    return normalize_counterfact_schema(list(payload))


def filter_counterfact_with_gpt2(
    items: list[dict[str, str]],
    model: Any,
    *,
    max_items: int = 5000,
) -> list[dict[str, Any]]:
    """Keep rows where GPT-2 greedy last-token pred matches ``target_true`` (single token)."""
    kept: list[dict[str, Any]] = []
    for ex in items[:max_items]:
        toks_true = model.to_tokens(" " + ex["target_true"].strip(), prepend_bos=False)
        toks_false = model.to_tokens(" " + ex["target_false"].strip(), prepend_bos=False)
        if toks_true.shape[1] != 1 or toks_false.shape[1] != 1:
            continue
        import torch  # noqa: PLC0415

        with torch.no_grad():
            logits = model(ex["prompt"])[0, -1]
        pred = int(logits.argmax().item())
        if pred == int(toks_true[0, 0].item()):
            kept.append(
                {
                    **ex,
                    "tok_true": int(toks_true[0, 0].item()),
                    "tok_false": int(toks_false[0, 0].item()),
                }
            )
    return kept


def save_counterfact_filtered(
    items: list[dict[str, Any]],
    data_dir: str | None = None,
) -> Path:
    out_path = filtered_json_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_counterfact_filtered(data_dir: str | None = None) -> list[dict[str, Any]]:
    path = filtered_json_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"CounterFact filtered JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
