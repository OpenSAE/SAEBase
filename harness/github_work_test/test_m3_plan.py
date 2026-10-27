"""M3-1 contract and planning tests, no model downloads."""
import copy
import pytest
from saebase.benchmarks import BenchmarkConfig, MetricResult, build_plan, get_metric

CONFIG = {
    "schema": "saebase.benchmark.config.v1", "name": "offline-test",
    "targets": [{"model_key": "synthetic", "sae_keys": ["sae-a", "sae-b"],
                 "layer": 0, "hook_name": "blocks.0.hook_resid_post"}],
    "dataset": {"id": "unit", "samples": [
        {"sample_id": "s1", "token_ids": [1, 2, 3], "prompt_group_id": "g1", "split": "test"},
        {"sample_id": "s2", "token_ids": [4, 5], "prompt_group_id": "g2", "split": "test"}]},
    "metrics": ["core.l0", "core.reconstruction_mse", "intervention.logit_delta_l2"],
    "conditions": [{"name": "identity", "operation": "identity"}],
}

def test_deterministic_plan():
    cfg = BenchmarkConfig.from_dict(CONFIG)
    assert BenchmarkConfig.from_dict(cfg.to_dict()).run_id == cfg.run_id
    plan = build_plan(cfg)
    assert plan["planned_measurements"] == 4
    assert plan["metric_jobs"] == 6
    assert any(node["operation"] == "m2_measure" for node in plan["nodes"])
    assert all(node["cache_status"] == "unverified" for node in plan["nodes"])

def test_preflight_rejects_leakage_and_bad_config():
    x = copy.deepcopy(CONFIG)
    x["dataset"]["samples"][1]["prompt_group_id"] = "g1"
    x["dataset"]["samples"][1]["split"] = "train"
    with pytest.raises(ValueError, match="leakage"):
        BenchmarkConfig.from_dict(x)
    x = copy.deepcopy(CONFIG)
    x["metrics"].append("nonsense")
    with pytest.raises(ValueError, match="unregistered"):
        BenchmarkConfig.from_dict(x)
    x = copy.deepcopy(CONFIG)
    x["max_measurements"] = 2
    with pytest.raises(ValueError, match="budget"):
        build_plan(BenchmarkConfig.from_dict(x))

def test_metric_result_rejects_nans_and_undefined():
    base = {"metric_id": "core.l0", "metric_version": "1", "protocol_id": "m3",
            "model_id": "m", "sae_id": "s", "site": {"layer": 0}, "dataset_id": "d",
            "split": "test", "seed": 0, "sample_unit": "token",
            "n_requested": 3, "n_valid": 3, "status": "completed", "value": 2.0}
    assert MetricResult(**base).to_dict()["value"] == 2.
    with pytest.raises(ValueError):
        MetricResult(**{**base, "value": float("nan")})
    with pytest.raises(ValueError):
        MetricResult(**{**base, "status": "undefined", "value": None})
    assert get_metric("core.fvu").direction == "lower"
