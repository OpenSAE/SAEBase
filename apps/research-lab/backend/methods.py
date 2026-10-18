"""CPU-only adapters to the *unchanged* original SAR, RSS and Localize estimators.

These routines need real saved measurements, never run the language model.
Run only with trusted, pinned local research packages; never expose a path-to-code
argument to a public API. JSON outputs are descriptive until an experiment
report has passed source/data/protocol review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from .core import canonical_json, digest, load_sites, has_complete_identity
from .ingest import _atomic_bytes, _raw_jsonl


def _original(source: Path, expected_sha: str, filename: str):
    if source.name != filename or not source.is_file():
        raise ValueError("Use the original trusted " + filename + " checkout")
    if len(expected_sha) != 64 or digest(source) != expected_sha:
        raise ValueError("Original estimator hash mismatch")
    key = "opensae_original_" + filename.replace(".", "_")
    spec = importlib.util.spec_from_file_location(key, str(source.resolve()))
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load estimator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_site(payload: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    if payload.get("source_type") != "real_checkpoint":
        raise ValueError("Only genuine measured activations may enter a real analysis index")
    site = next((s for s in load_sites(repo_root) if s["site_id"] == payload.get("site_id")), None)
    if site is None:
        raise ValueError("Unregistered model/SAE/site")
    if not has_complete_identity(site):
        raise ValueError(
            "Original evidence has no complete weight digests; new observation must be separately verified"
        )
    if (
        payload.get("model_sha256") != site["model_sha256"]
        or payload.get("sae_sha256") != site["sae_sha256"]
    ):
        raise ValueError("Analysis model/SAE weight digest mismatch")
    for field in ("prompt_group_id", "dataset_release", "protocol_id"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError("Missing analysis identity: " + field)
    return site


def _number_or_undefined(key: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {"metric": key, "value": None, "status": "undefined"}
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError("Unexpected estimator output for " + key) from e
    if not math.isfinite(value):
        return {"metric": key, "value": None, "status": "undefined"}
    return {"metric": key, "value": value, "status": "completed"}


def run_sar(
    payload: dict[str, Any], *, original: Path, expected_sha: str, repo_root: Path | None = None
) -> dict[str, Any]:
    site = _validated_site(payload, repo_root=repo_root)
    if payload.get("protocol_id") != "sar.original.k16.m384.fit256.v1":
        raise ValueError("SAR configuration must match approved standard protocol")
    import numpy as np

    module = _original(original, expected_sha, "sar.py")
    masks = np.asarray(payload["masks"], dtype=bool)
    values = np.asarray(payload["g"], dtype=float)
    fit = np.asarray(payload["fit"], dtype=int)
    held = np.asarray(payload["held"], dtype=int)
    if (
        masks.shape != (384, 16)
        or values.shape != (384,)
        or fit.shape != (256,)
        or held.shape != (128,)
    ):
        raise ValueError("SAR standard protocol requires 384 x 16 masks, fit256/held128")
    if payload.get("g_full") is None:
        raise ValueError("Full-coalition baseline missing")
    observed = module.sar(values, masks, fit, held, g_full=payload["g_full"])
    rows = [_number_or_undefined(key, value) for key, value in observed.items()]
    return {
        "schema": "opensae.analysis_record.v1",
        "kind": "sar",
        "source_type": "real_checkpoint",
        "site_id": site["site_id"],
        "prompt_group_id": payload["prompt_group_id"],
        "dataset_release": payload["dataset_release"],
        "protocol_id": payload["protocol_id"],
        "source_sha256": expected_sha,
        "results": rows,
        "status": "completed",
        "description": "Original SAR: coalition interaction residuals and held-out metrics",
    }


def run_rss(
    payload: dict[str, Any], *, original: Path, expected_sha: str, repo_root: Path | None = None
) -> dict[str, Any]:
    site = _validated_site(payload, repo_root=repo_root)
    if payload.get("protocol_id") != "rss.transform.v1":
        raise ValueError("Unrecognized RSS transformation protocol")
    import numpy as np

    module = _original(original, expected_sha, "rss.py")
    selected_ids = payload.get("selection_variant_ids")
    held_ids = payload.get("held_variant_ids")
    if (
        not isinstance(selected_ids, list)
        or not isinstance(held_ids, list)
        or len(selected_ids) != 5
        or len(held_ids) != 10
        or len(set(selected_ids + held_ids)) != 15
    ):
        raise ValueError("RSS must have five selection and ten disjoint held-out variants")
    delta = np.asarray(payload["delta_selection"], dtype=float)
    held = np.asarray(payload["delta_held"], dtype=float)
    if (
        delta.ndim != 2
        or held.ndim != 2
        or delta.shape[0] != 5
        or held.shape != (10, delta.shape[1])
    ):
        raise ValueError("RSS delta shape mismatch")
    support, direction, mean = module.rss(delta)
    rates = module.held_out_rate(held, support, direction)
    return {
        "schema": "opensae.analysis_record.v1",
        "kind": "rss",
        "source_type": "real_checkpoint",
        "site_id": site["site_id"],
        "prompt_group_id": payload["prompt_group_id"],
        "dataset_release": payload["dataset_release"],
        "protocol_id": payload["protocol_id"],
        "source_sha256": expected_sha,
        "status": "completed",
        "selection_variant_ids": selected_ids,
        "held_variant_ids": held_ids,
        "support_ids": support.tolist(),
        "directions": direction.tolist(),
        "results": [
            _number_or_undefined("mean_rss", mean),
            _number_or_undefined(
                "held_out_direction_rate", float(rates.mean()) if len(rates) else None
            ),
        ],
        "description": "Original RSS: selected feature support and held-out direction persistence",
    }


def run_localize(payload: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    site = _validated_site(payload, repo_root=repo_root)
    import numpy as np
    from sae_localize import exact_scope_curve  # separately installed original package

    factors = payload.get("factor_names")
    if not isinstance(factors, list) or len(factors) != 6 or len(set(factors)) != 6:
        raise ValueError("The platform v1 controlled grid requires six distinct factors")
    if payload.get("state_order") != list(range(64)):
        raise ValueError("Grid is not the full ordered 2^6 Boolean design")
    codes = np.asarray(payload["feature_codes"], dtype=float)
    if codes.ndim != 2 or codes.shape[0] != 64 or not np.isfinite(codes).all():
        raise ValueError("Localize requires a full finite 64-state grid")
    result = exact_scope_curve(codes, n_factors=6)
    target = 0.1 * result.total_variance
    return {
        "schema": "opensae.analysis_record.v1",
        "kind": "localization",
        "source_type": "real_checkpoint",
        "site_id": site["site_id"],
        "prompt_group_id": payload["prompt_group_id"],
        "dataset_release": payload["dataset_release"],
        "protocol_id": payload["protocol_id"],
        "source_sha256": str(payload.get("original_package_sha256", "")),
        "factor_names": factors,
        "status": "completed",
        "results": [
            _number_or_undefined("budget_at_90pct", result.budget(target)),
            _number_or_undefined("total_variance", result.total_variance),
        ],
        "normalized_curve": result.normalized_curve.tolist() if result.total_variance > 0 else None,
        "description": "Original optimal scope curve on a complete six-factor grid",
    }


def build_analysis_index(records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    if (output / "manifest.json").exists():
        raise FileExistsError("Analysis release already published")
    if any(
        row.get("schema") != "opensae.analysis_record.v1"
        or row.get("kind") not in ("sar", "rss", "localization")
        or row.get("source_type") != "real_checkpoint"
        or not row.get("source_sha256")
        for row in records
    ):
        raise ValueError("Unverified analysis record")
    data = canonical_json(records)
    manifest = {
        "schema": "opensae.analysis_index.v1",
        "files": {"index.json": hashlib.sha256(data).hexdigest()},
        "count": len(records),
    }
    _atomic_bytes(output / "index.json", data)
    _atomic_bytes(output / "manifest.json", canonical_json(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sar", "rss", "localization", "publish-index"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-sha")
    args = parser.parse_args(argv)
    if args.action == "publish-index":
        info = build_analysis_index(list(_raw_jsonl(args.input)), args.output)
    else:
        with args.input.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if args.action == "sar":
            if not args.source or not args.expected_sha:
                parser.error("SAR needs original source and SHA")
            info = run_sar(payload, original=args.source, expected_sha=args.expected_sha)
        elif args.action == "rss":
            if not args.source or not args.expected_sha:
                parser.error("RSS needs original source and SHA")
            info = run_rss(payload, original=args.source, expected_sha=args.expected_sha)
        else:
            info = run_localize(payload)
        _atomic_bytes(args.output, canonical_json(info))
    print(json.dumps(info, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
