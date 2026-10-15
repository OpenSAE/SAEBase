"""Deterministic M4 group/paired bootstrap tests: no model or network."""
import numpy as np
import pytest
from saebase.reliability import (
    AuditConfig,Observation,bootstrap_metric,paired_bootstrap,sample_size_curve,group_samples)

def obs(sae="sae-a",offset=0,split="test",n_groups=6):
    result=[]
    for i in range(n_groups):
        for variant in ("clean","wrapper"):
            result.append(Observation(
                model_id="toy",sae_id=sae,metric_id="intervention.logit_delta_l2",
                dataset_id="toy-data",split=split,prompt_group_id=f"prompt-{i}",
                sample_id=f"{i}-{variant}",value=float(i+offset+(variant=="wrapper")),
                measurement_ref=f"{sae}-{i}-{variant}"))
    return result

def cfg():
    return AuditConfig(evaluation_seeds=(0,1,2,3,4),bootstrap_replicates=100,
                       sample_sizes=(2,4,16))

def test_grouped_repeatability_and_variants_are_not_independent():
    rows=obs()
    assert len(group_samples(rows))==6
    result=bootstrap_metric(rows,cfg())
    assert result==bootstrap_metric(rows,cfg())
    assert result["status"]=="completed"
    assert result["n_groups"]==6 and result["n_observations"]==12
    assert result["n_valid"]==500
    assert result["interval"][0]<=result["point_estimate"]<=result["interval"][1]
    assert result["point_estimate"]==pytest.approx(3.)

def test_paired_same_group_draws_and_exact_constant_shift():
    a,b=obs(),obs("sae-b",offset=2)
    result=paired_bootstrap(a,b,cfg())
    assert result["status"]=="completed"
    assert result["point_estimate"]==pytest.approx(-2.)
    assert result["interval"]==pytest.approx([-2.,-2.])
    b[0]=Observation(**{**b[0].__dict__,"sample_id":"mismatched"})
    with pytest.raises(ValueError,match="sample/condition mismatch"):
        paired_bootstrap(a,b,cfg())

def test_one_group_undefined_and_mixed_split_rejected():
    assert bootstrap_metric(obs(n_groups=1),cfg())["status"]=="undefined"
    rows=obs()
    rows[0]=Observation(**{**rows[0].__dict__,"split":"calibration"})
    with pytest.raises(ValueError,match="cannot mix"):
        bootstrap_metric(rows,cfg())

def test_curve_clamps_available_groups_and_seed_independence():
    curve=sample_size_curve(obs(),cfg())
    assert {(r["actual_groups"]) for r in curve}=={2,4,6}
    assert len(curve)==15
    assert all(r["status"]=="completed" for r in curve)

def test_config_rejects_invalid_seed_or_ci():
    with pytest.raises(ValueError,match="between"):
        AuditConfig(confidence=1.2)
    with pytest.raises(ValueError,match="training and evaluation"):
        AuditConfig(seed_kind="training")
    with pytest.raises(ValueError,match="distinct"):
        AuditConfig(evaluation_seeds=(1,1))
    assert AuditConfig.from_dict(cfg().to_dict())==cfg()
