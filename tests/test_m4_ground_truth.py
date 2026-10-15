"""M4 synthetic ground truths. Original paper parity is opt-in in a separate CI."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pytest

from saebase.reliability.schema import AuditConfig
from saebase.reliability.ground_truth import (
    make_factors,fit_matching,evaluate_matching,reconstruction_mse)
from saebase.reliability.interaction_truth import exact_order_energy
from saebase.reliability.controls import check_coordinates
from saebase.reliability.discriminability import evaluate_degradation
from saebase.reliability.ground_truth_report import evaluate_ground_truth,append_ground_truth
from saebase.integrations import load_paper_module

@pytest.mark.parametrize("scenario",[
    "independent","correlated","hierarchical","superposition"])
def test_known_factors_holdout_matching_permutation_and_truth(scenario):
    truth=make_factors(scenario,seed=17)
    assert not set(truth.calibration_ids)&set(truth.test_ids)
    assert truth.dictionary.shape[1]==6
    assert truth.dictionary.shape[0]==(4 if scenario=="superposition" else 8)
    mapping=fit_matching(truth.calibration,truth.calibration)
    score=evaluate_matching(truth.test,truth.test,mapping)
    assert score["score"]==pytest.approx(1.,abs=1e-10)
    assert score["n_test"]==128
    assert reconstruction_mse(truth.test,truth.dictionary)==pytest.approx(0.)
    perm=np.array([4,1,3,0,5,2])
    new_matching=fit_matching(truth.calibration,truth.calibration[:,perm])
    new_score=evaluate_matching(truth.test,truth.test[:,perm],new_matching)
    assert new_score["score"]==pytest.approx(1.,abs=1e-10)
    degraded=truth.test+np.random.default_rng(19).normal(size=truth.test.shape)
    assert reconstruction_mse(truth.test,truth.dictionary,degraded)>1e-4

def test_fourier_truth_distinguishes_first_pair_and_third_order():
    one=exact_order_energy("first_order")
    two=exact_order_energy("pairwise")
    three=exact_order_energy("third_order")
    assert one[2]<1e-12 and one[3]<1e-12
    assert two[2]==pytest.approx(2.5**2,abs=1e-10)
    assert two[3]<1e-12
    assert three[3]==pytest.approx(3.5**2,abs=1e-10)

def test_feature_permutation_scale_and_null_intervention_controls():
    assert check_coordinates(seed=10)["status"]=="passed"

def test_synthetic_degradation_and_grouped_rank_stability():
    cfg=AuditConfig(evaluation_seeds=(0,1),bootstrap_replicates=40,
                    sample_sizes=(32,64,128),top_k=1)
    results=evaluate_degradation(cfg)
    assert set(results)=={"independent","correlated","hierarchical","superposition"}
    assert all(x["status"]=="passed" for x in results.values())
    for case in results.values():
        for trial in case["trials"]:
            assert trial["quality"]["oracle"]["feature_recovery"]==pytest.approx(1.,abs=1e-10)
            assert trial["paired"]["interval"][1]<0
            assert trial["ranking"]["mean_top_k_overlap"]==1.
            assert trial["ranking"]["ranking_reversal_frequency"]==0.

def _sources():
    sar=os.getenv("SAEBASE_SAR_SOURCE")
    rss=os.getenv("SAEBASE_RSS_SOURCE")
    if not sar or not rss:
        pytest.skip("set original SAR and RSS paths to run source parity")
    return sar,rss

def test_original_sar_rss_known_truth_and_invariance():
    sar,rss=_sources()
    from saebase.reliability.interaction_truth import verify_original_sar
    from saebase.reliability.rss_truth import verify_original_rss
    from saebase.reliability.controls import check_sar_negatives,check_rss_controls
    a=load_paper_module(sar,paper="sar")
    b=load_paper_module(rss,paper="rss")
    result=verify_original_sar(a)
    assert all(x["status"]=="passed" for x in result.values()),result
    assert verify_original_rss(b)["status"]=="passed"
    assert check_sar_negatives(a)["status"]=="passed"
    assert check_rss_controls(b)["status"]=="passed"

def test_complete_source_frozen_ground_truth_report():
    sar,rss=_sources()
    cfg=AuditConfig(evaluation_seeds=(0,1),bootstrap_replicates=40)
    report=evaluate_ground_truth(sar_source=sar,rss_source=rss,config=cfg)
    assert report["status"]=="passed",report
    assert all(len(item["trials"])==2 for item in report["factor_cases"].values())
    assert len(report["coalition_reseed_trials"])==2
    assert all(x["status"]=="passed" for x in report["coalition_reseed_trials"])
    assert len(report["rss_reseed_trials"])==2
    assert all(x["status"]=="passed" for x in report["rss_reseed_trials"])
    assert len(report["source_provenance"]["sar"]["sha256"])==64
    attached=append_ground_truth({"schema":"saebase.m4.audit.v1","status":"completed"},report)
    assert attached["status"]=="completed"
    assert attached["ground_truth"]["status"]=="passed"

def test_ground_truth_attachment_rejects_modified_paper_source(tmp_path):
    sar,rss=_sources()
    import json,shutil
    from saebase.reliability.report import write_audit,read_audit
    original_sar=tmp_path/"sar.py"
    original_rss=tmp_path/"rss.py"
    shutil.copyfile(sar,original_sar)
    shutil.copyfile(rss,original_rss)
    cfg=AuditConfig(evaluation_seeds=(0,1),bootstrap_replicates=40)
    truth=evaluate_ground_truth(sar_source=original_sar,rss_source=original_rss,config=cfg)
    assert truth["status"]=="passed"
    root=tmp_path/"m3-run"
    root.mkdir()
    (root/"metrics_manifest.json").write_text(json.dumps({
        "schema":"saebase.m3.results.v1","run_id":"m3-run",
        "sha256":"a"*64,"rows":0}))
    audit=append_ground_truth({
        "schema":"saebase.m4.audit.v1","status":"completed",
        "m3_run_id":"m3-run","metric_audits":[],
        "paired_comparisons":[],"rank_stability":[]},truth)
    write_audit(root,audit)
    assert read_audit(root)==audit
    original_sar.write_text(original_sar.read_text()+"\n# tampered source\n")
    with pytest.raises(ValueError,match="source changed"):
        read_audit(root)
