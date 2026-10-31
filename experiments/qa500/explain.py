"""Deterministic feature-explanation evidence from frozen QA500 activations.

Only source-reported descriptions are attached. Held-out top-feature occurrence
is a descriptive activity check, NOT semantic/casual explanation validation.
"""

from __future__ import annotations

from experiments.qa500.paths import task_dir, typed_file

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.qa500.cohort import PROTOCOL_ID, verify_dataset
from saebase.integrations.neuronpedia import load_annotations, load_mapping, annotate


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def explain_from_captures(
    *,
    data_root: Path,
    result_root: Path,
    model_key: str,
    sae_key: str,
    layer: int,
    dataset_id: str,
    mapping_file: Path | None = None,
    annotation_file: Path | None = None,
) -> dict[str, Any]:
    verify_dataset(dataset_id, data_root)
    site_id = f"{model_key}:{sae_key}:L{layer}"
    capture_dir = task_dir(
        result_root,
        task="capture",
        dataset=dataset_id,
        model=model_key,
        sae_key=sae_key,
        layer=layer,
    )
    base = task_dir(
        result_root,
        task="explain",
        dataset=dataset_id,
        model=model_key,
        sae_key=sae_key,
        layer=layer,
        initialize=True,
    )
    rows = _read(typed_file(capture_dir, "metrics", "capture.jsonl"))
    if len(rows) != 500 or len({r["sample_id"] for r in rows}) != 500:
        raise RuntimeError("Explanation discovery requires complete 500-sample capture")
    if any(row.get("site_id") != site_id or row.get("status") != "executed" for row in rows):
        raise RuntimeError("Foreign SAE site in explanation source")
    shas = {row["sae_sha256"] for row in rows}
    if len(shas) != 1:
        raise RuntimeError("Explanation source has mixed SAE weights")
    # Split by frozen ID, never by answer or feature identity.
    by_hash = sorted(
        rows,
        key=lambda r: hashlib.sha256(("explain:" + r["sample_id"]).encode()).hexdigest(),
    )
    discovery, heldout = by_hash[:100], by_hash[100:]
    frequency = Counter(f["feature_id"] for row in discovery for f in row["top_features"])
    if len(frequency) < 2:
        raise RuntimeError("Too few observed SAE features for fixed explanation budget")
    selected = sorted(frequency, key=lambda f: (-frequency[f], f))[:2]
    external_map = load_mapping(mapping_file)
    source_labels = load_annotations(annotation_file)
    candidates = []
    for feature_id in selected:
        source = annotate(
            feature_id,
            site_id=site_id,
            sae_sha256=next(iter(shas)),
            mapping=external_map,
            annotations=source_labels,
        )
        heldout_hits = sum(
            any(f["feature_id"] == feature_id for f in row["top_features"]) for row in heldout
        )
        candidates.append(
            {
                **source,
                "discovery_top10_count": frequency[feature_id],
                "discovery_n": len(discovery),
                "heldout_top10_count": heldout_hits,
                "heldout_n": len(heldout),
                "semantic_explanation_validated": False,
                "measurement": "top10_feature_occurrence_not_full_activation_frequency",
            }
        )
    output = {
        "protocol_id": PROTOCOL_ID,
        "task": "explain",
        "site_id": site_id,
        "dataset_id": dataset_id,
        "sae_sha256": next(iter(shas)),
        "status": "source_annotations_only",
        "source": "Neuronpedia public export or API if exact checkpoint mapped",
        "selection_rule": "top2_discovery_top10_frequency_v1",
        "candidates": candidates,
        "discovery_sample_ids_sha256": hashlib.sha256(
            json.dumps([r["sample_id"] for r in discovery]).encode()
        ).hexdigest(),
        "heldout_sample_ids_sha256": hashlib.sha256(
            json.dumps([r["sample_id"] for r in heldout]).encode()
        ).hexdigest(),
    }
    dest = typed_file(base, "reports", "explain.json")
    if dest.is_file():
        previous = json.loads(dest.read_text(encoding="utf-8"))
        if previous != output:
            # Changing annotation source counts as an intentional update.
            previous_path = typed_file(base, "reports", "explain_previous.json")
            if previous_path.exists():
                raise RuntimeError("Explanation revision requires a new run revision")
            previous_path.write_text(json.dumps(previous, indent=2) + "\n")
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output
