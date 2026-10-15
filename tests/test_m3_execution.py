"""M3 offline E2E, all public interfaces; no network or pretrained weights."""
import json
from types import SimpleNamespace
import pytest
import numpy as np
from examples._synthetic_common import ResearchToyModel,ResearchToySAE
from saebase.api import ModelAdapter
from saebase.benchmarks import BenchmarkConfig,build_plan
from saebase.benchmarks.runner import execute
from saebase.benchmarks.results import read_metrics
from saebase.benchmarks.adapters.saebench import import_native
from saebase.benchmarks.metrics.reconstruction import loss_recovered

BASE={
 "schema":"saebase.benchmark.config.v1","name":"synthetic-integration",
 "targets":[{"model_key":"synthetic","sae_keys":["toy"],"layer":0,
             "hook_name":"blocks.0.hook_resid_post"}],
 "dataset":{"id":"synthetic-samples","samples":[
    {"sample_id":"a","token_ids":[1,2,3],"prompt_group_id":"a","split":"test",
     "target_token_id":1,"reference_token_id":0},
    {"sample_id":"b","token_ids":[4,5,6],"prompt_group_id":"b","split":"test",
     "target_token_id":2,"reference_token_id":0}]},
 "metrics":["core.l0","core.reconstruction_mse","core.fvu","core.explained_variance",
            "intervention.logit_delta_l2","intervention.kl_baseline_to_edited",
            "intervention.edited_target_top1"],
 "conditions":[{"name":"identity","operation":"identity"},
               {"name":"reconstruction","operation":"reconstruction"}],
 "max_measurements":10,"max_activation_rows":20,
 "options":{"tokenizer_sha256":"b"*64},
}

def test_offline_benchmark_and_resume(tmp_path):
    cfg=BenchmarkConfig.from_dict(BASE)
    model=ModelAdapter(ResearchToyModel().eval())
    sae=ResearchToySAE().eval()
    calls={"n":0}
    def loader(model_key,sae_key,layer):
        calls["n"]+=1
        return SimpleNamespace(model=model,sae=sae)
    assert build_plan(cfg)["planned_measurements"]==4
    with pytest.raises(ValueError,match="real=True"):
        execute(cfg,tmp_path,loader=loader)
    assert calls["n"]==0
    rows=execute(cfg,tmp_path,loader=loader,real=True)
    assert calls["n"]==1
    assert len(rows)==4+3*2
    assert all(r["status"]=="completed" for r in rows)
    for row in rows:
        assert np.isfinite(row["value"])
    assert set(r["condition_id"] for r in rows if r["metric_id"].startswith("intervention."))=={"identity","reconstruction"}
    report=tmp_path/cfg.run_id/"report.html"
    assert report.exists()
    assert (tmp_path/cfg.run_id/"summary.csv").exists()
    assert (tmp_path/cfg.run_id/"metrics.jsonl").exists()
    again=execute(cfg,tmp_path,loader=loader,real=True,resume=True)
    assert again==rows
    assert calls["n"]==1
    assert len(read_metrics(tmp_path/cfg.run_id,cfg.run_id))==len(rows)
    with pytest.raises(FileExistsError):
        execute(cfg,tmp_path,loader=loader,real=True,resume=False)
    (tmp_path/cfg.run_id/"metrics.jsonl").write_text("{}\n")
    with pytest.raises(ValueError,match="integrity"):
        read_metrics(tmp_path/cfg.run_id,cfg.run_id)

def test_loss_recovered_and_invalid_denominator():
    assert loss_recovered(1.,2.,3.)==pytest.approx(0.5)
    assert loss_recovered(1.,2.,1.) is None

@pytest.mark.parametrize(("id","category","value"),[
    ("core","sparsity",14.5),
    ("sparse_probing","sae",0.7)])
def test_native_saebench_import_matches_original_value(tmp_path,id,category,value):
    # Contract fixture verifies faithful importing, NOT an actual upstream run.
    item={"eval_type_id":id,"sae_bench_commit_hash":"f"*40,
          "sae_lens_id":"toy","eval_result_metrics":{
             category:{"l0":value,"sae_top_1_test_accuracy":value}}}
    f=tmp_path/"upstream.json";f.write_text(json.dumps(item))
    metric="saebench.core.l0" if id=="core" else "saebench.sparse_probing.sae_top_1_test_accuracy"
    result,sha,commit=import_native(f,metric,expected_commit="f"*40,expected_sae_id="toy")
    assert result==value and len(sha)==64 and commit=="f"*40
    with pytest.raises(ValueError):
        import_native(f,metric,expected_commit="a"*40)

def test_changed_metric_config_reuses_m1_forward(tmp_path):
    """Changing a metric must not collect the same activation samples again."""
    import copy
    model=ModelAdapter(ResearchToyModel().eval())
    sae=ResearchToySAE().eval()
    counter={"n":0}
    # Count real hook-bearing model invocations, rather than calls to loader.
    orig=model.model.run_with_cache
    def counted(*args,**kwargs):
        counter["n"]+=1
        return orig(*args,**kwargs)
    model.model.run_with_cache=counted
    def loader(*args):
        return SimpleNamespace(model=model,sae=sae)
    cfg_data=copy.deepcopy(BASE)
    cfg_data["metrics"]=["core.l0"]
    first=BenchmarkConfig.from_dict(cfg_data)
    execute(first,tmp_path,loader=loader,real=True)
    count=counter["n"]
    assert count==len(first.samples)
    cfg_data["metrics"]=["core.l0","core.reconstruction_mse"]
    second=BenchmarkConfig.from_dict(cfg_data)
    assert first.run_id!=second.run_id
    execute(second,tmp_path,loader=loader,real=True)
    assert counter["n"]==count, "new metric must reuse existing M1 activation store"

def test_reconstruction_behavior_computes_measured_ablation(tmp_path):
    """Reconstruction and zero-site ablation use the exact same model/readout."""
    import copy
    data=copy.deepcopy(BASE)
    data["metrics"]=["reconstruction.target_nll","reconstruction.loss_recovered"]
    data["conditions"]=[]
    cfg=BenchmarkConfig.from_dict(data)
    model=ModelAdapter(ResearchToyModel().eval())
    sae=ResearchToySAE().eval()
    rows=execute(cfg,tmp_path,real=True,
                 loader=lambda *args: SimpleNamespace(model=model,sae=sae))
    assert len(rows)==2
    assert rows[0]["metric_id"]=="reconstruction.target_nll"
    assert rows[0]["status"]=="completed"
    assert np.isfinite(rows[0]["value"])
    assert rows[1]["metric_id"]=="reconstruction.loss_recovered"
    # Denominator zero is explicitly undefined, not converted to 0 or NaN.
    assert rows[1]["status"] in {"completed","undefined"}
    if rows[1]["status"]=="completed":
        assert np.isfinite(rows[1]["value"])

def test_intervention_preflight_requires_conditions():
    import copy
    data=copy.deepcopy(BASE)
    data["conditions"]=[]
    cfg=BenchmarkConfig.from_dict(data)
    with pytest.raises(ValueError,match="require explicit conditions"):
        build_plan(cfg)

def test_upstream_native_import_never_loads_model_and_rejects_stale_outputs(tmp_path):
    import copy
    data=copy.deepcopy(BASE)
    data["metrics"]=["saebench.core.l0"]
    data["conditions"]=[]
    source=tmp_path/"core_native.json"
    source.write_text(json.dumps({
        "eval_type_id":"core","sae_bench_commit_hash":"a"*40,
        "sae_lens_id":"toy-official","sae_lens_release_id":"toy-official-release",
        "eval_config":{"model_name":"synthetic-official"},
        "eval_result_metrics":{"sparsity":{"l0":7.5}},
    }))
    data["sources"]={"saebench.core.l0":str(source)}
    data["options"]={"saebench_sae_lens_ids":{"toy":"toy-official"},
                     "saebench_model_names":{"synthetic":"synthetic-official"},
                     "saebench_sae_releases":{"toy":"toy-official-release"},
                     "saebench_commit":"a"*40}
    cfg=BenchmarkConfig.from_dict(data)
    plan=build_plan(cfg)
    assert plan["planned_activation_rows"]==0
    assert not any(n["operation"].startswith("m1_") for n in plan["nodes"])
    def no_weights(*args):
        raise AssertionError("import-only benchmark must not download weights")
    rows=execute(cfg,tmp_path,real=True,loader=no_weights)
    assert len(rows)==1
    assert rows[0]["status"]=="completed" and rows[0]["value"]==7.5
    again=execute(cfg,tmp_path,real=True,loader=no_weights)
    assert again==rows
    source.write_text(source.read_text()+" ")
    with pytest.raises(ValueError,match="output changed"):
        execute(cfg,tmp_path,real=True,loader=no_weights)

def test_rss_metric_units_keep_selection_and_heldout_separate():
    import copy
    from saebase.benchmarks.runner import _metric
    from saebase.benchmarks.registry import get_metric
    data=copy.deepcopy(BASE)
    data["metrics"]=["rss.mean_rss","rss.held_out_persistence"]
    data["dataset"]["samples"]=[
        {"sample_id":"sel"+str(i),"token_ids":[1,2,3],"variant_role":"selection"}
        for i in range(3)] + [
        {"sample_id":"held"+str(i),"token_ids":[1,2,4],"variant_role":"held_out"}
        for i in range(5)]
    data["conditions"]=[]
    cfg=BenchmarkConfig.from_dict(data)
    t=cfg.targets[0]
    mean=_metric(cfg,t,"toy",get_metric("rss.mean_rss"),"completed",.8,n_valid=3)
    persistence=_metric(cfg,t,"toy",get_metric("rss.held_out_persistence"),"completed",.9,n_valid=5)
    assert mean.n_requested==mean.n_valid==3
    assert persistence.n_requested==persistence.n_valid==5

def test_new_registered_metric_without_executor_is_explicitly_unsupported(tmp_path):
    import copy
    from saebase.benchmarks.schema import MetricSpec
    from saebase.benchmarks.registry import register_metric,get_metric
    key="experimental.m3_no_executor"
    try:
        get_metric(key)
    except ValueError:
        register_metric(MetricSpec(key,"1",("external_fixture",),"dataset"))
    data=copy.deepcopy(BASE)
    data["metrics"]=[key]
    data["conditions"]=[]
    cfg=BenchmarkConfig.from_dict(data)
    model=ModelAdapter(ResearchToyModel().eval())
    sae=ResearchToySAE().eval()
    rows=execute(cfg,tmp_path,real=True,
                 loader=lambda *args: SimpleNamespace(model=model,sae=sae))
    assert len(rows)==1
    assert rows[0]["metric_id"]==key
    assert rows[0]["status"]=="unsupported"
    assert "no executable adapter" in rows[0]["reason"]
