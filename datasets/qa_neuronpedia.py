"""Versioned, exact-match Neuronpedia annotations, with no guessed feature IDs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

# Neuronpedia model IDs/SAE dictionary IDs must be explicitly verified against
# the *physical* checkpoint. Empty by default: a similar layer is not evidence.
APPROVED_MAPPINGS: dict[str, dict[str, str]] = {}


def load_mapping(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return APPROVED_MAPPINGS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Neuronpedia mapping must be a dictionary")
    for key, spec in data.items():
        if not isinstance(spec, dict) or any(
            not isinstance(spec.get(k), str) or not spec.get(k)
            for k in ("model", "source", "sae_sha256")
        ):
            raise ValueError(f"Invalid NP mapping for {key}")
        if len(spec["sae_sha256"]) != 64:
            raise ValueError("Neuronpedia mapping requires real SAE weight SHA256")
    return data


def annotate(
    feature_id: int, *, site_id: str, sae_sha256: str | None,
    mapping: dict[str, dict[str, str]], annotations: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Use independently downloaded feature data, never construct claims from IDs."""
    base: dict[str, Any] = {
        "feature_id": feature_id, "explanation": None,
        "explanation_status": "unmapped", "explanation_source": None,
    }
    cfg = mapping.get(site_id)
    if not cfg:
        return base
    if not sae_sha256 or sae_sha256 != cfg["sae_sha256"]:
        return {**base, "explanation_status": "checkpoint_mismatch"}
    key = f"{cfg['model']}/{cfg['source']}/{feature_id}"
    row = (annotations or {}).get(key)
    if row is None:
        return {**base, "explanation_status": "not_available"}
    if not isinstance(row, dict) or row.get("feature_id") != feature_id:
        raise ValueError("Neuronpedia annotation feature ID mismatch")
    if row.get("model") != cfg["model"] or row.get("source") != cfg["source"]:
        raise ValueError("Neuronpedia annotation model/SAE source mismatch")
    text = row.get("description")
    if not isinstance(text, str) or not text.strip():
        return {**base, "explanation_status": "no_description"}
    return {
        **base, "explanation": text.strip(), "explanation_status": "source_reported",
        "explanation_source": f"https://www.neuronpedia.org/{key}",
    }


def load_annotations(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = f"{obj['model']}/{obj['source']}/{obj['feature_id']}"
            if key in records:
                raise ValueError(f"Duplicate Neuronpedia annotation {key}")
            records[key] = obj
    return records
