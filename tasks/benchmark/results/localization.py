"""Purpose: Read localization curves, reconstruct frozen code grids and check their source and protocol identity.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from saebase.datasets.controlled_inputs import (
    ControlledInputManifest,
    load_controlled_input_manifest,
)

from saebase.store import fingerprint
from tasks.benchmark.methods.localization import (
    localization_grid_from_code_store,
)

SCHEMA = "saebase.localization.artifact.v1"

_METRICS = ("localization.budget_at_90pct", "localization.budget_fraction_at_90pct")

_SHA = re.compile(r"[0-9a-f]{64}\Z")

_REF = re.compile(r"localization/[0-9a-f]{32}\.json\Z")


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
    from tasks.benchmark.methods.localization import (
        estimator_source_sha256,
        parse_localization_options,
    )

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
        raise ValueError("Localize artifact differs from benchmark metric identity")
    if estimator_source_sha256(paper) != data["source_sha256"]:
        raise ValueError("original Localize estimator changed since benchmark execution")
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
