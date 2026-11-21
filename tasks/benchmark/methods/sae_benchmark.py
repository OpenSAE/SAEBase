"""Purpose: Import native SAEBench measurements with their model, SAE and upstream source identity.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tasks.benchmark.results.metrics import _metric


def _import_saebench(config, target, sae_key, specs):
    """Pure verified native result import: no SAE/model weights or forwards."""
    out = []
    expected_ids = config.options.get("saebench_sae_lens_ids", {})
    expected_models = config.options.get("saebench_model_names", {})
    expected_releases = config.options.get("saebench_sae_releases", {})
    for spec in specs:
        source = config.sources.get(spec.metric_id)
        if not source:
            out.append(
                _metric(
                    config,
                    target,
                    sae_key,
                    spec,
                    "blocked",
                    reason="native SAEBench output path required; no local upstream run executed",
                )
            )
            continue
        try:
            if not isinstance(expected_ids, dict) or not expected_ids.get(sae_key):
                raise ValueError(
                    "options.saebench_sae_lens_ids must identify the native SAE checkpoint"
                )
            if not isinstance(expected_models, dict) or not expected_models.get(target.model_key):
                raise ValueError("options.saebench_model_names must identify the native model")
            value, sha, commit = import_native(
                source,
                spec.metric_id,
                expected_commit=config.options.get("saebench_commit"),
                expected_sae_id=expected_ids[sae_key],
                expected_model_name=expected_models[target.model_key],
                expected_sae_release=expected_releases.get(sae_key),
            )
            out.append(
                _metric(
                    config,
                    target,
                    sae_key,
                    spec,
                    "completed",
                    value,
                    n_valid=1,
                    source_sha=sha,
                    source_commit=commit,
                )
            )
        except (ValueError, TypeError, KeyError, FileNotFoundError, OSError) as exc:
            out.append(
                _metric(
                    config, target, sae_key, spec, "blocked", reason=f"{type(exc).__name__}: {exc}"
                )
            )
    return out


PATHS = {
    "saebench.core.l0": (("sparsity", "l0"),),
    "saebench.core.mse": (("reconstruction_quality", "mse"),),
    "saebench.sparse_probing.sae_top_1_test_accuracy": (("sae", "sae_top_1_test_accuracy"),),
}


def import_native(
    path,
    metric_id,
    *,
    expected_commit=None,
    expected_sae_id=None,
    expected_model_name=None,
    expected_sae_release=None,
):
    if metric_id not in PATHS:
        raise ValueError("unsupported SAEBench metric")
    file = Path(path)
    content = file.read_bytes()
    item = json.loads(content)
    eval_id = "core" if metric_id.startswith("saebench.core.") else "sparse_probing"
    if item.get("eval_type_id") != eval_id:
        raise ValueError(f"expected native SAEBench eval_type_id {eval_id}")
    commit = item.get("sae_bench_commit_hash")
    if not isinstance(commit, str) or not commit.strip():
        raise ValueError("native SAE Bench commit provenance required")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("native SAEBench commit mismatch")
    if expected_sae_id is not None and item.get("sae_lens_id") != expected_sae_id:
        raise ValueError("native SAEBench SAE mismatch")
    if expected_sae_release is not None and item.get("sae_lens_release_id") != expected_sae_release:
        raise ValueError("native SAEBench SAE release mismatch")
    if expected_model_name is not None:
        conf = item.get("eval_config")
        if not isinstance(conf, dict) or conf.get("model_name") != expected_model_name:
            raise ValueError("native SAEBench model_name mismatch")
    metrics = item.get("eval_result_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("native SAEBench eval_result_metrics missing")
    path_keys = PATHS[metric_id][0]
    value = metrics
    for key in path_keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("missing native SAE Bench metric " + "/".join(path_keys))
        value = value[key]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("nonfinite/non-scalar native SAE Bench metric")
    return float(value), hashlib.sha256(content).hexdigest(), commit
