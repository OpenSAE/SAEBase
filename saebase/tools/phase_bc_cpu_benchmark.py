"""Genuine CPU-only M3 Benchmark and M4 Reliability acceptance for three Stage B pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from saebase.tools.phase_bc_cpu_pair import load_cpu_pair, PROMPTS
from saebase.api import capture_features
from saebase.benchmarks import BenchmarkConfig, build_plan
from saebase.benchmarks.runner import execute
from saebase.benchmarks.report import write_report
from saebase.reliability import AuditConfig, evaluate_m3, write_audit, read_audit


def run(sae_key: str, output: Path) -> dict:
    entry, profile, model, sae, meta, site, model_name, raw_model = load_cpu_pair(sae_key)
    first = capture_features(model, sae, site=site, prompt=PROMPTS[0])
    fid = int(torch.argmax(first.codes.abs()).item())
    if not torch.isfinite(first.codes).all() or float(first.codes[fid].abs()) == 0:
        raise ValueError("no real SAE feature activated for Benchmark")
    samples=[]
    for i,prompt in enumerate(PROMPTS):
        ids = model.tokens(prompt=prompt).detach().cpu().tolist()[0]
        samples.append({
            "sample_id": f"stage-bc-{entry.model_key}-{i}",
            "token_ids": ids, "split": "test",
            "prompt_group_id": f"stage-bc-group-{i}",
            "selection": {"mode":"indices","indices":[len(ids)-1]},
        })
    config=BenchmarkConfig.from_dict({
        "schema":"saebase.benchmark.config.v1",
        "name":f"Stage B real CPU SAEBase Benchmark {sae_key}",
        "targets":[{
            "model_key":entry.model_key, "sae_keys":[sae_key],
            "layer":site.layer,"hook_name":site.hook_name,
        }],
        "dataset":{"id":f"stage-bc-real-{entry.model_key}","samples":samples},
        "metrics":[
            "core.l0", "core.reconstruction_mse", "core.explained_variance",
            "intervention.logit_delta_l2",
        ],
        "conditions":[
            {"name":"identity","operation":"identity"},
            {"name":"active_feature_ablation","operation":"ablate","feature_ids":[fid]},
        ],
        "max_activation_rows":16, "max_measurements":24,
        "sample_batch":1, "microbatch":1, "seed":0, "tolerance":.01,
    })
    rows = execute(
        config,output,real=True,
        loader=lambda mk,sk,l: SimpleNamespace(model=model,sae=sae),
    )
    if len(rows)!=5 or any(x["status"]!="completed" for x in rows):
        raise ValueError("real Stage B Benchmark incomplete")
    root=output/config.run_id
    audit_cfg=AuditConfig(
        evaluation_seeds=(0,1,2), bootstrap_replicates=32,
        sample_sizes=(2,3), metric_ids=("core.l0","intervention.logit_delta_l2"),
    )
    audit=evaluate_m3(root,audit_cfg)
    if len(audit["metric_audits"])!=3 or any(x["status"]!="completed" for x in audit["metric_audits"]):
        raise ValueError("Stage B M4 missing native metric audit")
    write_audit(root,audit)
    write_report(root,config,build_plan(config),rows)
    if read_audit(root)!=audit:
        raise ValueError("Stage B M4 integrity mismatch")
    record={
        "schema":"saebase.coverage.cpu_pair_m3m4.v1",
        "status":"benchmark_and_reliability_verified",
        "sae_key":sae_key,"model_key":entry.model_key,
        "model_repo":model_name,"sae_repo":profile.repo_id or profile.sae_release,
        "layer":site.layer,"feature_ablated":fid,
        "sample_count":len(samples),"metric_rows":len(rows),
        "reliability_audits":len(audit["metric_audits"]),
        "model_forwards_in_m4":0,"m3_run_id":config.run_id,
        "metric_results":[{
            "metric_id":x["metric_id"],"condition":x["condition_id"],
            "value":x["value"],"n_valid":x["n_valid"],
        } for x in rows],
        "m3_sha256":json.loads((root/"metrics_manifest.json").read_text())["sha256"],
        "m4_sha256":json.loads((root/"reliability_manifest.json").read_text())["audit_sha256"],
    }
    (output/f"{sae_key}_m3m4_summary.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-key",required=True)
    p.add_argument("--output",type=Path,default=Path("stage-bc-m3m4"))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(run(args.sae_key,args.output),indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
