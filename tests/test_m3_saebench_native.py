"""Verify official native SAEBench benchmark values were imported unchanged."""
import json
import os
from pathlib import Path
import pytest

from saebase.benchmarks.adapters.saebench import import_native

ROOT = os.getenv("SAEBASE_SAEBENCH_ROOT")
if not ROOT:
    pytest.skip("requires pinned official SAEBench repository", allow_module_level=True)

@pytest.mark.parametrize("metric,relpath,expected", [
    ("saebench.core.l0",
     "tests/acceptance/test_data/core/core_expected_results.json",
     376.97979736328125),
    ("saebench.core.mse",
     "tests/acceptance/test_data/core/core_expected_results.json",
     0.002910538576543331),
    ("saebench.sparse_probing.sae_top_1_test_accuracy",
     "tests/acceptance/test_data/sparse_probing/sparse_probing_expected_results.json",
     0.7744),
])
def test_official_fixture_value_imported_exactly(metric, relpath, expected):
    """Historical upstream reference fixture import, not a new native run."""
    path = Path(ROOT) / relpath
    original = json.loads(path.read_text())
    value, digest, commit = import_native(
        path, metric, expected_commit=original["sae_bench_commit_hash"],
        expected_sae_id=original["sae_lens_id"])
    assert value == pytest.approx(expected, abs=1e-10)
    assert digest and len(digest) == 64
    assert commit == original["sae_bench_commit_hash"]

def test_bounded_real_native_core_and_sparse_probing_same_checkpoint():
    """New official upstream runs; pinned model, SAE checkpoint, dataset, config."""
    raw = os.getenv("SAEBASE_NATIVE_RESULTS")
    if not raw:
        pytest.skip("native smoke outputs not configured")
    root = Path(raw)
    evidence = json.loads((root / "evidence.json").read_text())
    assert evidence["status"] == "passed"
    assert evidence["upstream_commit"] == "8042bb3828c6340da8d12062324e92b2077c571c"
    for prefix in ("core", "sparse_probing"):
        path = root / evidence[prefix + "_file"]
        native = json.loads(path.read_text())
        assert native["eval_config"]["model_name"] == evidence["model_name"]
        assert native["sae_lens_id"] == evidence["sae_id"]
        assert native["sae_lens_release_id"] == evidence["sae_release"]
        assert native["eval_type_id"] == prefix
    for metric, expected in evidence["native_metrics"].items():
        kind = "core" if metric.startswith("saebench.core.") else "sparse_probing"
        path = root / evidence[kind + "_file"]
        imported, sha, commit = import_native(
            path, metric,
            expected_commit=evidence["upstream_commit"],
            expected_sae_id=evidence["sae_id"],
            expected_model_name=evidence["model_name"],
            expected_sae_release=evidence["sae_release"])
        assert imported == expected
        assert sha and len(sha) == 64
        assert commit == evidence["upstream_commit"]
