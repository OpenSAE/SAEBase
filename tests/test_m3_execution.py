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
