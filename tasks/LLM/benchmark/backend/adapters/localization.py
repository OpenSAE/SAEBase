"""Localize M3 adapter: validated full input grids and checksum-bound curve artifacts.

Exact/paired scope estimators remain in the independent sae_localize package.
Metric rows contain scalar summaries; full curves and assignments are artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from tasks.LLM.benchmark.controlled_inputs import (
    BinaryFactor,
    ControlledInputManifest,
    MAX_FACTORS,
    load_controlled_input_manifest,
    save_controlled_input_manifest,
)
from tasks.LLM.benchmark.integrations.localization import (
    evaluate_localization,
    localization_grid_from_code_store,
    _original_localize,
)
from saebase.store import ShardedReader, fingerprint, resolve_positions
from tasks.LLM.benchmark.backend.results import atomic_json


SCHEMA = "saebase.localization.artifact.v1"
_METRICS = ("localization.budget_at_90pct", "localization.budget_fraction_at_90pct")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REF = re.compile(r"localization/[0-9a-f]{32}\.json\Z")


@dataclass(frozen=True)
class LocalizationOptions:
    name: str
    factors: tuple[BinaryFactor, ...]
    sample_ids: tuple[str, ...]
    pairs_per_scope: int | None = None

    @property
    def n_factors(self) -> int:
        return len(self.factors)


def parse_localization_options(config: Any) -> LocalizationOptions:
    """Preflight a complete single-grid experiment, without loading weights."""
    raw = config.options.get("localization")
    if not isinstance(raw, dict) or set(raw) - {
        "name",
        "factors",
        "sample_ids_by_state",
        "pairs_per_scope",
    }:
        raise ValueError("options.localization needs name, factors and sample_ids_by_state")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("options.localization.name must be nonempty")
    factors_raw = raw.get("factors")
    if not isinstance(factors_raw, list) or not 1 <= len(factors_raw) <= MAX_FACTORS:
        raise ValueError(f"localization factors must contain 1..{MAX_FACTORS} binary factors")
    factors = []
    for item in factors_raw:
        if not isinstance(item, dict) or set(item) != {"name", "levels"}:
            raise ValueError("localization factor requires name and two levels")
        if not isinstance(item["levels"], list):
            raise ValueError("localization factor levels must be a list")
        factors.append(BinaryFactor(item["name"], tuple(item["levels"])))
    factors = tuple(factors)
    if len({f.name for f in factors}) != len(factors):
        raise ValueError("localization factor names must be unique")
    order = raw.get("sample_ids_by_state")
    if (
        not isinstance(order, list)
        or len(order) != 1 << len(factors)
        or any(not isinstance(x, str) or not x.strip() for x in order)
        or len(set(order)) != len(order)
    ):
        raise ValueError("localization sample_ids_by_state must enumerate every state exactly once")
    samples = {s.record.sample_id: s.record for s in config.samples}
    if set(order) != set(samples):
        raise ValueError("localization state samples must exactly match the benchmark dataset")
    if len({s.split for s in samples.values()}) != 1:
        raise ValueError("localization requires one evaluation split per complete grid")
    if any(s.variant_role != "controlled" for s in samples.values()):
        raise ValueError("localization samples must have variant_role='controlled'")
    groups = {s.prompt_group_id for s in samples.values()}
    if len(groups) != 1 or None in groups or "" in groups:
        raise ValueError(
            "one complete controlled input grid is one prompt group, not independent states"
        )
    token_keys = set()
    for sid in order:
        sample = samples[sid]
        if len(resolve_positions(sample)) != 1:
            raise ValueError("localization needs exactly one selected token position per state")
        if sample.token_ids in token_keys:
            raise ValueError("two controlled states have identical token IDs")
        token_keys.add(sample.token_ids)
    pairs = raw.get("pairs_per_scope")
    if pairs is not None and (type(pairs) is not int or pairs < 1):
        raise ValueError("localization pairs_per_scope must be a positive integer")
    return LocalizationOptions(name, factors, tuple(order), pairs)


def manifest_from_config(
    config: Any, options: LocalizationOptions, activation_identity_key: str
) -> ControlledInputManifest:
    """Derive a checked P2 sidecar from already-tokenized M3 SampleRecords."""
    lookup = {s.record.sample_id: s.record for s in config.samples}
    rows = []
    for mask, sid in enumerate(options.sample_ids):
        record = lookup[sid]
        positions = resolve_positions(record)
        rows.append(
            {
                "mask": mask,
                "levels": [(mask >> i) & 1 for i in range(options.n_factors)],
                "sample_id": sid,
                "position": positions[0],
                "token_site_sha256": fingerprint(
                    {
                        "token_ids": list(record.token_ids),
                        "positions": list(positions),
                    }
                ),
            }
        )
    design_key = fingerprint(
        {
            "schema": "saebase.localization.config_design.v1",
            "name": options.name,
            "factors": [f.to_dict() for f in options.factors],
            "states": [
                {"sample_id": r["sample_id"], "token_site_sha256": r["token_site_sha256"]}
                for r in rows
            ],
        }
    )
    manifest = ControlledInputManifest(
        name=options.name,
        design_key=design_key,
        factors=options.factors,
        states=tuple(rows),
        activation_identity_key=activation_identity_key,
    )
    return ControlledInputManifest.from_dict(manifest.to_dict())


def estimator_source_sha256(paper: Any | None = None) -> str:
    """Hash the exact original estimator Python source used, never a copied implementation."""
    original = _original_localize(paper)
    func = original.exact_scope_curve
    code = getattr(func, "__code__", None)
    if code is None or not Path(code.co_filename).is_file():
        raise ValueError("Localize estimator source is not a readable Python file")
    return hashlib.sha256(Path(code.co_filename).read_bytes()).hexdigest()


def _artifact_reference(config: Any, target: Any, sae_key: str) -> str:
    return (
        "localization/"
        + fingerprint(
            {
                "run_id": config.run_id,
                "model": target.model_key,
                "site": {
                    "layer": target.site.layer,
                    "hook": target.site.hook_name,
                    "position": target.site.position,
                },
                "sae": sae_key,
            }
        )[:32]
        + ".json"
    )


def run_localization(
    config: Any,
    target: Any,
    sae_key: str,
    *,
    activation_store: str | Path,
    code_store: str | Path,
    root: str | Path,
    paper: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute the native estimator against verified M1 caches and save full curves."""
    options = parse_localization_options(config)
    activation_store, code_store, root = (Path(x) for x in (activation_store, code_store, root))
    activation = ShardedReader(activation_store, verify_shards=True)
    code = ShardedReader(code_store, verify_shards=True)
    if code.kind != "codes" or activation.kind != "activations":
        raise ValueError("Localize requires SAEBase activation and code caches")
    if code_store.parent.parent.resolve() != activation_store.resolve():
        raise ValueError("Localize code cache does not belong to requested activation store")
    if not _SHA.fullmatch(activation_store.name) or not _SHA.fullmatch(code_store.name):
        raise ValueError("Localize cache paths must have SHA-256 identity keys")
    manifest = manifest_from_config(config, options, activation.manifest["identity_key"])
    save_controlled_input_manifest(
        activation_store / "controlled_inputs" / f"{manifest.design_key}.json", manifest
    )
    grid = localization_grid_from_code_store(code_store, manifest=manifest, site=target.site)
    if grid.state_sample_ids != options.sample_ids:
        raise ValueError("Localize code grid does not match benchmark state order")
    result = evaluate_localization(
        grid,
        paper=paper,
        relative_omission=0.1,
        pairs_per_scope=options.pairs_per_scope,
        seed=config.seed,
    )
    sha = estimator_source_sha256(paper)
    if result.exact_source_sha256 != sha:
        raise ValueError("Localize estimator source changed while evaluating the grid")
    budget_fraction = result.budget / (grid.n_factors * grid.n_features)
    data = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "model_id": target.model_key,
        "sae_id": sae_key,
        "site": {
            "layer": target.site.layer,
            "hook_name": target.site.hook_name,
            "position": target.site.position,
        },
        "dataset_id": config.dataset_id,
        "split": config.samples[0].record.split,
        "source_sha256": sha,
        "estimator_version": result.estimator_version,
        "design_manifest": manifest.to_dict(),
        "cache": {"activation_key": activation_store.name, "code_key": code_store.name},
        "n_features": grid.n_features,
        "n_factors": grid.n_factors,
        "total_variance": result.total_variance,
        "relative_omission": 0.1,
        "budget_at_90pct": result.budget,
        "budget_fraction_at_90pct": budget_fraction,
        "curve": result.curve.tolist(),
        "normalized_curve": (
            result.normalized_curve.tolist() if result.normalized_curve is not None else None
        ),
        "assignment": [int(x) for x in result.assignment],
        "pairs_per_scope": options.pairs_per_scope,
        "paired_seed": config.seed if options.pairs_per_scope is not None else None,
        "paired_estimated_curve": (
            result.estimated_curve.tolist() if result.estimated_curve is not None else None
        ),
        "interpretation": "complete_uniform_binary_product_grid",
    }
    data["artifact_sha256"] = fingerprint(data)
    ref = _artifact_reference(config, target, sae_key)
    dest = root / ref
    if dest.exists() and read_localization_artifact(root, ref) != data:
        raise ValueError("Localize artifact differs from existing frozen result; choose a new run")
    atomic_json(dest, data)
    return data, ref


def read_localization_artifact(root: str | Path, ref: str) -> dict[str, Any]:
    """Resolve a trusted run-local reference and validate the complete curve payload."""
    if not isinstance(ref, str) or not _REF.fullmatch(ref):
        raise ValueError("invalid Localize run-relative artifact reference")
    data = json.loads((Path(root) / ref).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("invalid Localize artifact schema")
    saved = data.get("artifact_sha256")
    payload = {k: v for k, v in data.items() if k != "artifact_sha256"}
    if not isinstance(saved, str) or fingerprint(payload) != saved:
        raise ValueError("Localize artifact integrity mismatch")
    manifest = load_controlled_input_manifest(
        ControlledInputManifest.from_dict(data["design_manifest"])
    )
    nf = data["n_factors"]
    m = data["n_features"]
    if type(nf) is not int or type(m) is not int or nf < 1 or m < 1 or nf != manifest.n_factors:
        raise ValueError("Localize artifact dimensions invalid")
    expected = nf * m + 1
    curve = np.asarray(data["curve"], dtype=np.float64)
    if curve.shape != (expected,) or not np.isfinite(curve).all():
        raise ValueError("Localize stored curve has invalid shape/values")
    W = data["total_variance"]
    b = data["budget_at_90pct"]
    f = data["budget_fraction_at_90pct"]
    if (
        type(W) not in (int, float)
        or not math.isfinite(W)
        or W < 0
        or type(b) is not int
        or not 0 <= b < expected
        or type(f) not in (float, int)
        or not math.isfinite(f)
        or not math.isclose(f, b / (nf * m), abs_tol=1e-12)
    ):
        raise ValueError("Localize budget/variance invalid")
    if data["relative_omission"] != 0.1:
        raise ValueError("Localize named 90% metric has inconsistent tolerance")
    if curve[b] > 0.1 * W + 1e-9 or (b > 0 and curve[b - 1] <= 0.1 * W - 1e-9):
        raise ValueError("Localize budget not consistent with optimal curve")
    assignments = data["assignment"]
    if (
        not isinstance(assignments, list)
        or len(assignments) != m
        or any(type(x) is not int or not 0 <= x < 1 << nf for x in assignments)
        or sum(x.bit_count() for x in assignments) > b
    ):
        raise ValueError("invalid Localize scope assignment")
    normal = data["normalized_curve"]
    if W == 0:
        if normal is not None:
            raise ValueError("constant Localize grid must not have normalized curve")
    else:
        norm = np.asarray(normal, dtype=np.float64)
        if norm.shape != curve.shape or not np.allclose(norm, curve / W, atol=1e-10, rtol=1e-10):
            raise ValueError("inconsistent normalized Localize curve")
    paired = data["paired_estimated_curve"]
    if paired is not None:
        values = np.asarray(paired, dtype=np.float64)
        if values.shape != curve.shape or not np.isfinite(values).all():
            raise ValueError("invalid Localize paired curve")
    for key in ("activation_key", "code_key"):
        if not isinstance(data["cache"].get(key), str) or not _SHA.fullmatch(data["cache"][key]):
            raise ValueError("invalid Localize source cache identity key")
    return data


def validate_localization_result(
    root: str | Path,
    config: Any,
    row: dict[str, Any],
    *,
    paper: Any | None = None,
) -> dict[str, Any]:
    """Check metric scalar, run, design and original source against the artifact."""
    refs = row.get("measurement_refs", [])
    if not isinstance(refs, list) or len(refs) != 1:
        raise ValueError("Localize completed metric requires one curve artifact reference")
    data = read_localization_artifact(root, refs[0])
    if (
        data["run_id"] != config.run_id
        or data["model_id"] != row["model_id"]
        or data["sae_id"] != row["sae_id"]
        or data["site"] != row["site"]
        or data["dataset_id"] != row["dataset_id"]
        or data["split"] != row["split"]
        or data["source_sha256"] != row.get("source_sha256")
    ):
        raise ValueError("Localize artifact differs from M3 metric identity")
    if estimator_source_sha256(paper) != data["source_sha256"]:
        raise ValueError("original Localize estimator changed since M3 execution")
    options = parse_localization_options(config)
    if options.sample_ids != tuple(
        data["design_manifest"]["states"][i]["sample_id"] for i in range(1 << options.n_factors)
    ):
        raise ValueError("Localize artifact input-state mapping changed")
    if options.pairs_per_scope != data["pairs_per_scope"]:
        raise ValueError("Localize artifact paired sampling protocol changed")
    value = (
        data["budget_at_90pct"]
        if row["metric_id"] == _METRICS[0]
        else data["budget_fraction_at_90pct"]
    )
    if row["metric_id"] not in _METRICS:
        raise ValueError("unknown Localize benchmark metric")
    if not math.isclose(row["value"], value, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("Localize metric scalar disagrees with stored exact curve")
    if row["n_valid"] != 1 or row["sample_unit"] != "dataset":
        raise ValueError("Localize metric unit differs from one complete grid")
    return data


def grid_from_artifact(root: str | Path, artifact: dict[str, Any]):
    """Read the exact frozen SAE code grid, without model forward or rescoring."""
    cache = artifact["cache"]
    root = Path(root)
    path = root.parent / "_cache" / "m1" / cache["activation_key"] / "codes" / cache["code_key"]
    manifest = ControlledInputManifest.from_dict(artifact["design_manifest"])
    grid = localization_grid_from_code_store(
        path,
        manifest=manifest,
        site=__import__("saebase.api", fromlist=["ActivationSite"]).ActivationSite(
            **artifact["site"]
        ),
    )
    if grid.n_features != artifact["n_features"] or grid.design_key != manifest.design_key:
        raise ValueError("Localize source codes do not match stored scope curve identity")
    return grid
