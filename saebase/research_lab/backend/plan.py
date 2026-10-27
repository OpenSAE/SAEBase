"""Deterministic CPU work-plan compiler for the GPU handoff.

It can run *before* dataset downloads. Missing inputs result in blocked jobs,
never guessed weights, unlicensed prompt text, or synthetic measurements.

The resulting JSON is a manifest of intended tasks, not evidence that
the tasks ran or a schedule that pays for GPU instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .core import (
    PLATFORM,
    canonical_json,
    data_root,
    dataset_catalog,
    digest,
    has_complete_identity,
    load_published_dataset,
    load_sites,
    read_json,
)
from .ingest import _atomic_bytes

COVERAGE_PATH = PLATFORM / "data" / "coverage.json"
SCHEMA = "opensae.compute_plan.v1"


def _job_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()[:24]


def _dataset_releases(root: Path | None = None) -> list[dict[str, Any]]:
    directory = (root or data_root()) / "datasets"
    sources = sorted(dataset_catalog()["sources"], key=lambda s: s["id"])
    releases = []
    for source in sources:
        manifest_path = directory / source["id"] / "manifest.json"
        has_manifest = manifest_path.is_file()
        records = None
        try:
            records = load_published_dataset(source["id"], root)
        except Exception:
            # An invalid published manifest must not be promoted to ready.
            pass
        ready = records is not None
        releases.append(
            {
                "dataset_id": source["id"],
                "ready": ready,
                "count": len(records) if ready else 0,
                "manifest_sha256": digest(manifest_path) if ready else None,
                "reason": None
                if ready
                else (
                    "unreviewed_or_invalid_release"
                    if has_manifest
                    else "source_not_collected_or_reviewed"
                ),
            }
        )
    return releases


def build_compute_plan(
    *,
    root: Path | None = None,
    coverage_path: Path = COVERAGE_PATH,
) -> dict[str, Any]:
    coverage = read_json(coverage_path)
    if coverage.get("schema") != "opensae.platform.coverage.v1":
        raise ValueError("Unrecognized coverage contract")
    sites = sorted(load_sites(), key=lambda s: s["site_id"])
    site_index = {site["site_id"]: site for site in sites}
    if len(sites) != 48 or len(site_index) != 48:
        raise ValueError("Expected immutable 48-site historical reference matrix")
    deep = coverage["deep"]
    sar = coverage["sar"]
    if (
        len(deep) != 6
        or len(set(deep)) != 6
        or len(sar) != 3
        or len(set(sar)) != 3
        or not set(sar) <= set(deep)
        or not set(deep) <= set(site_index)
        or coverage["online_candidate"] not in set(deep)
    ):
        raise ValueError("Inconsistent deep-analysis coverage")
    datasets = _dataset_releases(root)
    if len(datasets) != 10 or sum(s["count"] for s in datasets) not in range(0, 501):
        raise ValueError("Expected ten curated source datasets")
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        for task, targets in (
            ("capture", sites),
            ("intervention", [site_index[s] for s in deep]),
            ("sar", [site_index[s] for s in sar]),
            ("rss", [site_index[s] for s in deep]),
        ):
            for site in targets:
                identity_ready = has_complete_identity(site)
                reasons = []
                if not dataset["ready"]:
                    reasons.append(dataset["reason"])
                if not identity_ready:
                    reasons.append("historical_site_missing_full_weight_hash")
                if task == "intervention":
                    reasons.append("feature_and_readout_selection_pending")
                if task == "sar":
                    reasons.append("coalition_masks_and_scalar_readouts_pending")
                if task == "rss":
                    reasons.append("validated_aligned_variant_manifest_pending")
                spec = {
                    "schema": "opensae.compute_job.v1",
                    "task": task,
                    "model_key": site["model"],
                    "sae_key": site["sae"],
                    "site_id": site["site_id"],
                    "layer": site["layer"],
                    "hook_name": site["hook"],
                    "model_sha256": site["model_sha256"],
                    "sae_sha256": site["sae_sha256"],
                    "dataset_id": dataset["dataset_id"],
                    "dataset_manifest_sha256": dataset["manifest_sha256"],
                    "n_base_groups": dataset["count"] if dataset["ready"] else None,
                    "protocol_id": coverage["protocols"][task],
                    "independent_unit": "original_prompt_group",
                    "max_coalitions_per_group": coverage["coalition_per_prompt"]
                    if task == "sar"
                    else None,
                    "source_data_class": "curated_10x50_not_historical_stage_d",
                }
                jobs.append(
                    {
                        "job_id": _job_id(spec),
                        "spec": spec,
                        "status": "ready_for_external_gpu_operator" if not reasons else "blocked",
                        "blockers": reasons,
                        "execution": "not_run",
                    }
                )
    for design_name in coverage["localize_designs"]:
        spec = {
            "schema": "opensae.compute_job.v1",
            "task": "localization",
            "design": design_name,
            "factor_count": 6,
            "n_factor_states": 64,
            "protocol_id": coverage["protocols"]["localization"],
            "source_data_class": "independent_controlled_grid_not_500_natural_prompts",
        }
        jobs.append(
            {
                "job_id": _job_id(spec),
                "spec": spec,
                "status": "blocked",
                "blockers": ["frozen_full_factor_grid_and_identity_pending"],
                "execution": "not_run",
            }
        )
    statuses: dict[str, int] = {}
    for job in jobs:
        statuses[job["status"]] = statuses.get(job["status"], 0) + 1
    return {
        "schema": SCHEMA,
        "status": "planning_only",
        "generated_from": {
            "coverage_contract_sha256": digest(coverage_path),
            "historical_reference": "conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json",
        },
        "datasets": datasets,
        "site_count": len(sites),
        "site_count_with_full_weight_hashes": sum(has_complete_identity(site) for site in sites),
        "job_counts": {
            "capture": len(sites) * len(datasets),
            "intervention": len(deep) * len(datasets),
            "sar": len(sar) * len(datasets),
            "rss": len(deep) * len(datasets),
            "localization": len(coverage["localize_designs"]),
        },
        "statuses": statuses,
        "estimated_sar_coalition_measurements_if_all_frozen": (
            len(sar) * 500 * coverage["coalition_per_prompt"]
        ),
        "estimate_note": "Logical coalition evaluations only; excludes controls, prompts, retries and token costs",
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write versioned plan as JSON")
    args = parser.parse_args(argv)
    payload = build_compute_plan()
    if args.output:
        if args.output.exists():
            raise FileExistsError("Compute plan is immutable; use a new output path")
        _atomic_bytes(args.output, canonical_json(payload))
    else:
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in (
                        "schema",
                        "status",
                        "site_count",
                        "site_count_with_full_weight_hashes",
                        "job_counts",
                        "statuses",
                        "estimated_sar_coalition_measurements_if_all_frozen",
                    )
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
