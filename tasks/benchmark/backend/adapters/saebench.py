"""Strict importer for native SAEBench Core / Sparse Probing JSON outputs.

Does not reproduce upstream math. Native SAEBench must run independently with
matching checkpoints, samples and protocol before validated comparison claims.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

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
