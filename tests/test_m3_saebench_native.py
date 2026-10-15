"""Pinned official SAEBench native reference results, without math reimplementation.

The external workflow first runs two *original* upstream evaluations, then
this test imports their results and verifies exact numerical ingestion.
"""
import os
from pathlib import Path
import json
import pytest
from saebase.benchmarks.adapters.saebench import import_native

ROOT=os.getenv("SAEBASE_SAEBENCH_ROOT")
if not ROOT:
    pytest.skip("requires checked-out official SAEBench repository",allow_module_level=True)

@pytest.mark.parametrize("metric,relative,tol,relpath,expected",[
 ("saebench.core.l0","core",0.02,
  "tests/acceptance/test_data/core/core_expected_results.json",376.97979736328125),
 ("saebench.core.mse","core",0.02,
  "tests/acceptance/test_data/core/core_expected_results.json",0.002910538576543331),
 ("saebench.sparse_probing.sae_top_1_test_accuracy","sparse_probing",0.04,
  "tests/acceptance/test_data/sparse_probing/sparse_probing_expected_results.json",0.7744),
])
def test_official_native_fixture_exact_import(metric,relative,tol,relpath,expected):
    upstream=Path(ROOT)/relpath
    item=json.loads(upstream.read_text())
    actual,digest,commit=import_native(upstream,metric,
                          expected_commit=item["sae_bench_commit_hash"],
                          expected_sae_id=item["sae_lens_id"])
    assert actual==pytest.approx(expected,abs=1e-10)
    assert len(digest)==64
    assert commit==item["sae_bench_commit_hash"]

def test_native_core_and_sparse_outputs_exist_and_match_reference():
    """Validates presence/protocol of newly generated original upstream results."""
    folder=Path(ROOT)
    core=list((folder/"tests/acceptance/test_data/core").glob("*_eval_results.json"))
    sparse=list((folder/"evals/sparse_probing/test_results").glob("*_eval_results.json"))
    assert core, "native Core eval did not produce a result"
    assert sparse, "native Sparse Probing eval did not produce a result"
    for path,metric,reference_path,tolerance in (
        (core[0],"saebench.core.l0",
         folder/"tests/acceptance/test_data/core/core_expected_results.json",0.02),
        (sparse[0],"saebench.sparse_probing.sae_top_1_test_accuracy",
         folder/"tests/acceptance/test_data/sparse_probing/sparse_probing_expected_results.json",0.04),
    ):
        native=json.loads(path.read_text())
        expected=json.loads(reference_path.read_text())
        assert native["sae_lens_id"]==expected["sae_lens_id"]
        value,digest,commit=import_native(path,metric,
            expected_commit=native["sae_bench_commit_hash"],
            expected_sae_id=expected["sae_lens_id"])
        reference,_,_=import_native(reference_path,metric)
        assert value==pytest.approx(reference,abs=tolerance)
        assert len(digest)==64
