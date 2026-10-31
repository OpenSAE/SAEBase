"""QA500 source-only annotation aggregation and Neuronpedia provenance sync."""
from __future__ import annotations


# --- explain ---
"""Deterministic feature-explanation evidence from frozen QA500 activations.

Only source-reported descriptions are attached. Held-out top-feature occurrence
is a descriptive activity check, NOT semantic/casual explanation validation.
"""


from scripts.experiments.qa500.context import task_dir, typed_file

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.experiments.qa500.cohort import PROTOCOL_ID, verify_dataset
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


# --- neuronpedia_sync ---
"""Neuronpedia bulk-export ingestion for QA500 without rate-limited feature scraping.

Official public S3 exports are preferable to hundreds of thousands of
individual feature API requests. Annotations are always contingent on a
user-auditable exact mapping to the loaded SAE weight digest.
"""


from scripts.experiments.qa500.context import task_dir, typed_file, study_dir
from scripts.experiments.qa500.context import get_dataset_ids

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from saebase.integrations.neuronpedia import load_mapping
from scripts.experiments.qa500.capture import site_list
from scripts.experiments.qa500.cohort import PROTOCOL_ID


BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
MAX_BATCH_FILES_PER_SITE = 1000


def _urlopen(url: str, timeout: int = 60) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != (
        "neuronpedia-datasets.s3.us-east-1.amazonaws.com"
    ):
        raise ValueError("Only the documented Neuronpedia public export host is allowed")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Neuronpedia export HTTP {resp.status}")
        return resp.read()


def list_exports(prefix: str) -> list[str]:
    """List public S3 object keys in the fixed Neuronpedia v1 export prefix."""
    if not prefix.startswith("v1/") or ".." in prefix or "//" in prefix:
        raise ValueError("Invalid Neuronpedia export prefix")
    token = None
    keys: list[str] = []
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = BUCKET + "/?" + urllib.parse.urlencode(params)
        xml = ET.fromstring(_urlopen(url))
        for content in xml.findall("{*}Contents"):
            elem = content.find("{*}Key")
            if elem is not None and elem.text:
                keys.append(elem.text)
        truncated = xml.findtext("{*}IsTruncated") == "true"
        if len(keys) > MAX_BATCH_FILES_PER_SITE:
            raise RuntimeError("Neuronpedia export contains too many batches")
        if not truncated:
            break
        token = xml.findtext("{*}NextContinuationToken")
        if not token:
            raise RuntimeError("S3 listing truncated without a continuation token")
    return [key for key in keys if key.endswith(".jsonl.gz") and "/explanations/" in key]


def _description(record: dict[str, Any]) -> tuple[int, str] | None:
    """Interpret documented export variants conservatively."""
    ident = record.get("index", record.get("featureIndex"))
    if ident is None and isinstance(record.get("feature"), dict):
        ident = record["feature"].get("index")
    try:
        index = int(ident)
    except (TypeError, ValueError):
        return None
    explanation = record.get("description")
    if not explanation and isinstance(record.get("explanations"), list):
        for item in record["explanations"]:
            if isinstance(item, dict) and isinstance(item.get("description"), str):
                explanation = item["description"]
                break
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    return index, explanation.strip()


def _capture_identity(
    results_root: Path,
    site: dict[str, Any],
) -> tuple[str, set[int]]:
    identities: set[str] = set()
    features: set[int] = set()
    for dataset_id in get_dataset_ids():
        run = task_dir(
            results_root,
            task="capture",
            dataset=dataset_id,
            model=site["model_key"],
            sae_key=site["sae_key"],
            layer=site["layer"],
        )
        dataset_file = typed_file(run, "metrics", "capture.jsonl")
        if not dataset_file.is_file():
            continue
        for line in dataset_file.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("site_id") != site["site_id"]:
                raise RuntimeError("Neuronpedia source capture has wrong SAE site")
            identities.add(row["sae_sha256"])
            features.update(int(f["feature_id"]) for f in row.get("top_features", []))
    if len(identities) != 1:
        raise RuntimeError("Source site has no complete single checkpoint identity")
    return next(iter(identities)), features


def sync(
    *,
    mapping_path: Path,
    result_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    mappings = load_mapping(mapping_path)
    sites = {site["site_id"]: site for site in site_list()}
    records: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {}
    for site_id in sorted(sites):
        if site_id not in mappings:
            report[site_id] = {"status": "unmapped"}
            continue
        cfg = mappings[site_id]
        digest, observed = _capture_identity(result_root, sites[site_id])
        if digest != cfg["sae_sha256"]:
            report[site_id] = {"status": "checkpoint_mismatch"}
            continue
        model = cfg["model"]
        source = cfg["source"]
        if any("/" in value or ".." in value for value in (model, source)):
            raise ValueError("Unsafe Neuronpedia source key")
        prefix = f"v1/{model}/{source}/explanations/"
        try:
            batch_keys = list_exports(prefix)
            if not batch_keys:
                report[site_id] = {"status": "no_public_explanations_export"}
                continue
            found = 0
            for key in batch_keys:
                blob = _urlopen(BUCKET + "/" + urllib.parse.quote(key, safe="/"))
                for line in gzip.decompress(blob).decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    value = _description(entry)
                    if value is None or value[0] not in observed:
                        continue
                    idx, explanation = value
                    rec_key = f"{model}/{source}/{idx}"
                    records[rec_key] = {
                        "model": model,
                        "source": source,
                        "feature_id": idx,
                        "description": explanation,
                    }
                    found += 1
            report[site_id] = {
                "status": "source_loaded",
                "observed_features": len(observed),
                "descriptions_available": found,
                "exports_used": len(batch_keys),
                "sae_sha256": digest,
            }
        except (ValueError, RuntimeError, OSError, ET.ParseError) as exc:
            report[site_id] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "reason": str(exc)[:300],
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(value, sort_keys=True, ensure_ascii=False)
        for _, value in sorted(records.items())
    ]
    data = "\n".join(lines) + ("\n" if lines else "")
    output_path.write_text(data, encoding="utf-8")
    summary = {
        "protocol_id": PROTOCOL_ID,
        "sites": report,
        "annotations": len(records),
        "output": str(output_path),
        "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
        "provenance": BUCKET,
    }
    (output_path.parent / "neuronpedia_sync.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Bulk Neuronpedia QA feature explanations")
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--result-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--output",
        type=Path,
        default=study_dir(Path.cwd()) / "metadata" / "neuronpedia" / "annotations.jsonl",
    )
    args = ap.parse_args(argv)
    print(
        json.dumps(
            sync(mapping_path=args.mapping, result_root=args.result_root, output_path=args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

