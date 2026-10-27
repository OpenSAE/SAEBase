"""M5 real-weight matrix, M1/M2/M3/M4 reuse, verified restart and metrics.

No external checkpoints are bundled, no score is collapsed across metrics,
and import-only/report-only modes never load model weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import resource
from typing import Any,Callable

from saebase.api import ModelAdapter,SAEIdentity
from tasks.LLM.benchmark.backend import BenchmarkConfig
from tasks.LLM.benchmark.backend.runner import execute as execute_benchmark
from tasks.LLM.benchmark.backend.results import atomic_json,read_metrics
from tasks.LLM.benchmark.backend.report import write_report
from tasks.LLM.benchmark.backend.planner import build_plan
from tasks.LLM.benchmark.reliability import AuditConfig
from tasks.LLM.benchmark.reliability.runner import evaluate_m3
from tasks.LLM.benchmark.reliability.report import write_audit,read_audit
from saebase.sae.registry import get_profile
from saebase.store import fingerprint
from .schema import MatrixConfig,MatrixTarget,STATUS_SCHEMA
from .planner import plan_matrix

def _sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def _dump(data:dict)->bytes:
    return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()

def _packages():
    result={"python":platform.python_version()}
    for name in ("torch","numpy","transformer-lens","sae-lens","transformers"):
        try:
            result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name]=None
    return result

def _default_model_loader(target:MatrixTarget,device:str):
    from saebase.sae.loading import load_model_for_profile
    raw,loaded_name=load_model_for_profile(
        target.sae_keys[0],device=device,model_key_override=target.model_key)
    return ModelAdapter(raw),loaded_name

def _default_sae_loader(target:MatrixTarget,key:str,device:str):
    from saebase.sae.loading import load_sae_for_profile
    sae,meta=load_sae_for_profile(key,target.site.layer,device=device)
    identity=SAEIdentity.from_loaded(sae=sae,metadata=meta,
                                      model_key=target.model_key)
    if identity.hook_name!=target.site.hook_name:
        raise ValueError(f"loaded SAE hook mismatch: {key}: {identity.hook_name}")
    return sae

def _prompt_samples(model:ModelAdapter,matrix:MatrixConfig,target:MatrixTarget):
    result=[]
    for idx,prompt in enumerate(matrix.prompts):
        token_ids=model.tokens(prompt=prompt)[0].detach().cpu().tolist()
        if not token_ids or len(token_ids)>matrix.max_token_count:
            raise ValueError(f"{target.model_key} prompt {idx} outside token count budget")
        result.append({"sample_id":f"m5-prompt-{idx:03d}",
            "token_ids":token_ids,"prompt_group_id":f"shared-prompt-{idx:03d}",
            "split":"test","readout_position":-1})
    return result

def _make_m3(matrix:MatrixConfig,target:MatrixTarget,samples,*,synthetic:bool=False):
    return BenchmarkConfig.from_dict({
        "schema":"saebase.benchmark.config.v1",
        "name":f"M5 {matrix.version}: {target.model_key}, {len(target.sae_keys)} SAEs",
        "targets":[{"model_key":target.model_key,"sae_keys":list(target.sae_keys),
                    "layer":target.site.layer,"hook_name":target.site.hook_name,
                    "position":target.site.position}],
        "dataset":{"id":f"saebase-m5-{matrix.matrix_id}-{target.model_key}",
                   "samples":samples},
        "metrics":list(matrix.metrics),"conditions":list(matrix.conditions),
        "seed":matrix.seed,"max_measurements":matrix.max_measurements,
        "max_activation_rows":len(samples),
        "sample_batch":1,"microbatch":2,"tolerance":0.01,
        "options":{"tokenizer_sha256":_sha(("synthetic-tokenizer:"+target.model_key).encode())}
                  if synthetic else {},
    })

def _peak_rss_bytes():
    # Linux GitHub-hosted runners report ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024

def _m3_summary(rows,target:MatrixTarget,matrix:MatrixConfig,provenance):
    included={key:[] for key in target.sae_keys}
    for row in rows:
        if row["sae_id"] not in included or row["status"]!="completed":
            raise ValueError(f"incomplete or unexpected M3 result: {row}")
        included[row["sae_id"]].append(row)
    planned=len(matrix.metrics)+sum(
        bool(key.startswith("intervention."))*(len(matrix.conditions)-1)
        for key in matrix.metrics)
    for key,items in included.items():
        if len(items)!=planned:
            raise ValueError(f"{key}: expected {planned} condition-aware metric rows, got {len(items)}")
    # Actual weight hashes establish independent checkpoints, not just names.
    hashes=[]
    for key in target.sae_keys:
        prefix=f"{target.model_key}/{target.site.layer}/{key}"
        item=provenance.get(prefix)
        if not item or len(item.get("sae_state_sha256",""))!=64:
            raise ValueError(f"missing exact SAE checkpoint fingerprint for {key}")
        hashes.append(item["sae_state_sha256"])
    if len(set(hashes))!=len(hashes):
        raise ValueError("M5 SAE checkpoint keys resolve to identical weights")
    # Every SAE after the first must reuse the model's exact M1 activations.
    for key in target.sae_keys[1:]:
        core=next(x for x in included[key] if x["metric_id"]=="core.l0")
        costs=core.get("cost",{})
        if costs.get("activations_collected")!=0 or costs.get("activations_reused")!=len(matrix.prompts):
            raise RuntimeError(f"M1 activation cache reuse failed for {target.model_key}/{key}")
    return {key:{"weight_sha256":hashes[i],
                "sae_release":get_profile(key).sae_release,
                "sae_id":get_profile(key).sae_id(target.site.layer),
                "sae_card":target.sae_cards[key],
                "metrics":len(included[key]),
                "cache_cost":next(x["cost"] for x in included[key] if x["metric_id"]=="core.l0")}
            for i,key in enumerate(target.sae_keys)}

def _write_report(root:Path,record:dict):
    rows=[]
    for family in record["models"]:
        for key,meta in family["checkpoints"].items():
            rows.append({"model":family["model_key"],"family":family["family"],
                "sae_key":key,"weight_sha256":meta["weight_sha256"],
                "metrics":meta["metrics"],"m3_run_id":family["m3_run_id"],
                "audit_groups":family["audit_groups"],
                "runtime_seconds":family["elapsed_seconds"]})
    with (root/"matrix_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    names=("model","family","sae_key","metrics","audit_groups","runtime_seconds")
    head="".join(f"<th>{html.escape(x)}</th>" for x in names)
    body="".join("<tr>"+"".join("<td>"+html.escape(str(r[k]))+"</td>" for k in names)+"</tr>" for r in rows)
    text=("<!doctype html><html><meta charset='utf-8'><title>SAEBase 0.2.0 Matrix</title>"
          "<style>body{font:14px system-ui;margin:3rem}table{border-collapse:collapse}"
          "td,th{padding:.5rem;border:1px solid #aaa}</style>"
          "<h1>SAEBase 0.2.0 Real Checkpoint Matrix</h1>"
          f"<p>Matrix ID: {html.escape(record['matrix_id'])}</p>"
          f"<p>Validated families: {len(record['models'])}; "
          f"independent checkpoints: {len(rows)}.</p>"
          "<p>Each metric has its own unit and interpretation. "
          "No composite SAE quality score is calculated.</p>"
          f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
          "<p>See each model's M3 and M4 reports for numerical estimates, "
          "uncertainty, conditions and complete SHA-256 provenance.</p></html>")
    (root/"matrix_report.html").write_text(text,encoding="utf-8")

def verify_matrix(root:str|Path,*,expected_id:str|None=None)->dict:
    root=Path(root)
    meta=json.loads((root/"matrix_manifest.json").read_text())
    raw=(root/"matrix_results.json").read_bytes()
    if meta.get("schema")!="saebase.m5.manifest.v1" or _sha(raw)!=meta.get("sha256"):
        raise ValueError("M5 result checksum/manifest mismatch")
    record=json.loads(raw)
    if record.get("schema")!=STATUS_SCHEMA or record.get("status")!="passed":
        raise ValueError("M5 matrix is not validated successfully")
    if expected_id is not None and record.get("matrix_id")!=expected_id:
        raise ValueError("M5 matrix identity mismatch")
    if meta.get("matrix_id")!=record.get("matrix_id"):
        raise ValueError("M5 result identity changed")
    if len({r["family"] for r in record["models"]})<2:
        raise ValueError("M5 requires two independent model families")
    if any(len(r["checkpoints"])<2 for r in record["models"]):
        raise ValueError("M5 requires at least two validated SAE checkpoints per model")
    root_real=root/"benchmarks"
    for family in record["models"]:
        path=root_real/family["m3_run_id"]
        m3=read_metrics(path,family["m3_run_id"])
        audit=read_audit(path)
        if audit["m3_run_id"]!=family["m3_run_id"]:
            raise ValueError("M5 M4 audit provenance mismatch")
        if any(x["status"]!="completed" for x in m3):
            raise ValueError("M5 child benchmark includes failed/unsupported metrics")
        provenance=json.loads((path/"provenance.json").read_text())
        for key,meta_sae in family["checkpoints"].items():
            pref=f"{family['model_key']}/{family['site']['layer']}/{key}"
            if provenance.get(pref,{}).get("sae_state_sha256")!=meta_sae["weight_sha256"]:
                raise ValueError("M5 recorded checkpoint hash disagrees with M3 provenance")
    return record

def execute_matrix(matrix:MatrixConfig,output:str|Path,*,device:str="cpu",
                   real:bool=False,resume:bool=True,
                   model_loader:Callable|None=None,sae_loader:Callable|None=None,
                   fail_after_models:int|None=None)->dict:
    plan=plan_matrix(matrix)
    if not real:
        raise ValueError("M5 checkpoint evaluation requires explicit real=True")
    root=Path(output)/matrix.matrix_id
    root.mkdir(parents=True,exist_ok=True)
    if (root/"matrix_manifest.json").exists():
        if not resume:
            raise FileExistsError("existing M5 matrix; use --resume or a fresh output directory")
        return verify_matrix(root,expected_id=matrix.matrix_id)
    models=[]
    started=perf_counter()
    test_loader=model_loader or _default_model_loader
    sae_getter=sae_loader or _default_sae_loader
    for target in matrix.targets:
        if perf_counter()-started>matrix.max_runtime_seconds:
            raise TimeoutError("M5 total elapsed runtime budget exceeded")
        model_started=perf_counter()
        model,loaded_name=test_loader(target,device)
        if not isinstance(model,ModelAdapter):
            model=ModelAdapter(model)
        if not model.capabilities.capture or not model.capabilities.patch:
            raise ValueError("M5 requires real capture and patch capable backend")
        samples=_prompt_samples(model,matrix,target)
        m3=_make_m3(matrix,target,samples,synthetic=model_loader is not None)
        cached_saes={}
        def get_sae(key):
            if key not in cached_saes:
                cached_saes[key]=sae_getter(target,key,device)
            return cached_saes[key]
        def pair_loader(model_key,sae_key,layer):
            if model_key!=target.model_key or layer!=target.site.layer:
                raise ValueError("M5 target/SAE mismatch")
            return (model,get_sae(sae_key))
        benchroot=root/"benchmarks"
        rows=execute_benchmark(m3,benchroot,real=True,resume=True,loader=pair_loader)
        path=benchroot/m3.run_id
        provenance=json.loads((path/"provenance.json").read_text())
        checkpoints=_m3_summary(rows,target,matrix,provenance)
        audit_cfg=AuditConfig(evaluation_seeds=(0,1,2,3,4),
            bootstrap_replicates=matrix.bootstrap_replicates,
            sample_sizes=tuple(sorted({2,4,len(samples)})),top_k=1)
        audit=evaluate_m3(path,audit_cfg,
            sae_resolver=lambda key,sae,layer:get_sae(sae))
        if audit["status"]!="completed" or any(x["status"]!="completed" for x in audit["metric_audits"]):
            raise RuntimeError("M5 requires full M4 real audit, with no undefined/blocked entries")
        write_audit(path,audit)
        write_report(path,m3,build_plan(m3),rows,provenance=provenance)
        if not (path/"report.html").exists() or "M4 Reliability Audit" not in (path/"report.html").read_text():
            raise RuntimeError("missing combined M3/M4 benchmark report")
        entry={"model_key":target.model_key,"family":target.family,
            "model_card":target.model_card,"model_loaded_name":loaded_name,
            "usage_note":target.usage_note,"site":target.to_dict()["site"],
            "m3_run_id":m3.run_id,"n_prompts":len(samples),
            "audit_groups":len(samples),"checkpoints":checkpoints,
            "elapsed_seconds":round(perf_counter()-model_started,5),
            "peak_process_rss_bytes":_peak_rss_bytes(),
            "audit_metric_rows":len(audit["metric_audits"]),
            "paired_comparisons":len(audit["paired_comparisons"])}
        models.append(entry)
        atomic_json(root/"matrix_progress.json",{
            "schema":"saebase.m5.progress.v1","matrix_id":matrix.matrix_id,
            "completed_model_keys":[m["model_key"] for m in models],
            "last_family":target.family,"status":"incomplete"})
        if fail_after_models is not None and len(models)>=fail_after_models:
            raise RuntimeError("injected M5 model-family interruption; rerun with resume")
    record={"schema":STATUS_SCHEMA,"matrix_id":matrix.matrix_id,
        "status":"passed","release_version":matrix.version,
        "run_config_sha256":fingerprint(matrix.to_dict()),
        "families":sorted({r["family"] for r in models}),
        "n_models":len(models),"n_distinct_saes":sum(len(x["checkpoints"]) for x in models),
        "n_prompts_per_model":len(matrix.prompts),
        "conditions":[c["name"] for c in matrix.conditions],
        "metric_ids":list(matrix.metrics),"models":models,
        "real_checkpoints":model_loader is None and sae_loader is None,
        "environment":_packages(),"total_runtime_seconds":round(perf_counter()-started,5),
        "peak_process_rss_bytes":_peak_rss_bytes(),
        "notes":"CPU/GPU results are not universal SAE rankings; no external weights redistributed."}
    if not record["real_checkpoints"]:
        record["status"]="synthetic_test"
        # Synthetic test runs must never satisfy the real release gate.
    raw=_dump(record)
    atomic_json(root/"matrix_results.json",record)
    atomic_json(root/"matrix_manifest.json",{
        "schema":"saebase.m5.manifest.v1","matrix_id":matrix.matrix_id,
        "sha256":_sha(raw)})
    if record["status"]=="passed":
        _write_report(root,record)
        verify_matrix(root,expected_id=matrix.matrix_id)
    return record
