"""M3 execution: M1 activation/code reuse, M2 measurements, metric adapters.

No model is loaded until `real=True`; no external source is downloaded.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from saebase.api import ModelAdapter,ActivationSite,load_registered
from saebase.interventions import BatchEngine,ExperimentIdentity,ResultWriter,RunBudget
from saebase.store import (ActivationIdentity,ActivationWriter,CodeIdentity,
                           ShardedReader,collect_activations,encode_from_store,sha256_state_dict)
from .schema import BenchmarkConfig, MetricResult
from .registry import get_metric
from .planner import build_plan
from .metrics.core import calculate_core
from .metrics.intervention import calculate_interventions
from .metrics.reconstruction import target_nll, loss_recovered
from .adapters.papers import run_paper
from .adapters.saebench import import_native
from .results import store_results,read_metrics
from .report import write_report

def _identity(config,target,model,sae):
    model_sha=sha256_state_dict(model.model)
    sae_sha=sha256_state_dict(sae)
    tokenizer_hash=config.options.get("tokenizer_sha256")
    if tokenizer_hash is None:
        tokenizer=getattr(model.model,"tokenizer",None)
        if tokenizer is None or not callable(getattr(tokenizer,"get_vocab",None)):
            raise ValueError("exact tokenizer identity required: set options.tokenizer_sha256")
        payload={"vocab":tokenizer.get_vocab(),
                 "special_tokens_map":getattr(tokenizer,"special_tokens_map",{}),
                 "added_vocab":tokenizer.get_added_vocab() if callable(getattr(tokenizer,"get_added_vocab",None)) else {}}
        tokenizer_hash=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,
                separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
    if not isinstance(tokenizer_hash,str) or len(tokenizer_hash)!=64 or any(ch not in "0123456789abcdef" for ch in tokenizer_hash):
        raise ValueError("options.tokenizer_sha256 must be lowercase SHA-256 hex")
    act=ActivationIdentity(model_key=target.model_key,model_state_sha256=model_sha,
            tokenizer_sha256=tokenizer_hash,hook_name=target.site.hook_name,layer=target.site.layer,
            preprocessing={"dataset":config.dataset_id,"site_position":target.site.position})
    return act,sae_sha

def _metric(config,target,sae_key,spec,status,value=None,n_valid=0,
            reason=None,condition=None,refs=(),source_sha=None,source_commit=None,cost=None):
    return MetricResult(metric_id=spec.metric_id,metric_version=spec.version,
        protocol_id=spec.protocol_id,model_id=target.model_key,sae_id=sae_key,
        site={"layer":target.site.layer,"hook_name":target.site.hook_name,
              "position":target.site.position},dataset_id=config.dataset_id,
        split="mixed" if len({s.record.split for s in config.samples})>1 else config.samples[0].record.split,
        seed=config.seed,sample_unit=spec.sample_unit,n_requested=len(config.samples),
        n_valid=n_valid,status=status,value=value,condition_id=condition,
        reason=reason,measurement_refs=tuple(refs),source_sha256=source_sha,
        source_commit=source_commit,cost=cost or {})

def _safe(config,target,sae_key,spec,fn,*,condition=None):
    try:
        payload=fn()
        value,n,refs,sha,commit=payload
        if n<=0 or value is None or not math.isfinite(float(value)):
            return _metric(config,target,sae_key,spec,"undefined",
                    reason="metric undefined or no valid observations",condition=condition)
        return _metric(config,target,sae_key,spec,"completed",float(value),
                       n_valid=n,condition=condition,refs=refs,
                       source_sha=sha,source_commit=commit)
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,IndexError,ZeroDivisionError) as exc:
        return _metric(config,target,sae_key,spec,"blocked",
                reason=f"{type(exc).__name__}: {exc}",condition=condition)

def execute(config: BenchmarkConfig, output: str|Path, *, real: bool=False,
            resume: bool=True, loader=None):
    """Execute a frozen matrix. A caller may inject a pair loader for offline tests.

    For real weights, caller explicitly passes real=True; failure to resolve an
    external adapter produces a blocked MetricResult rather than silent zeros.
    """
    plan=build_plan(config)
    if not real:
        raise ValueError("execution must be explicitly enabled with real=True")
    root=Path(output)/config.run_id
    root.mkdir(parents=True,exist_ok=True)
    if (root/"metrics_manifest.json").exists() and resume:
        prior=read_metrics(root,config.run_id)
        # A completed ledger is read-only. This never reexecutes model work.
        return prior
    if (root/"metrics_manifest.json").exists() and not resume:
        raise FileExistsError(f"existing benchmark results: {root}")
    results=[]
    provenance={}
    specs=[get_metric(m) for m in config.metrics]
    load=loader or (lambda model_key,sae_key,layer:
                    load_registered(model_key=model_key,sae_key=sae_key,layer=layer))
    cache_root=root/"m1_cache"
    for target in config.targets:
        for sae_key in target.sae_keys:
            pair=load(target.model_key,sae_key,target.site.layer)
            model=pair.model if hasattr(pair,"model") else pair[0]
            sae=pair.sae if hasattr(pair,"sae") else pair[1]
            if not isinstance(model,ModelAdapter):
                model=ModelAdapter(model)
            act_id,sae_sha=_identity(config,target,model,sae)
            identity=ExperimentIdentity(act_id.model_state_sha256,sae_sha,target.site,
                 model_key=target.model_key,sae_key=sae_key,
                 tokenizer_sha256=act_id.tokenizer_sha256,backend=model.backend)
            prefix=f"{target.model_key}/{target.site.layer}/{sae_key}"
            provenance[prefix]={"model_state_sha256":act_id.model_state_sha256,
                        "sae_state_sha256":sae_sha,
                        "tokenizer_sha256":act_id.tokenizer_sha256}
            records=[s.record for s in config.samples]
            wants_core=any(s.metric_id.startswith("core.") for s in specs)
            wants_m2=any(s.metric_id.startswith("intervention.") for s in specs)
            wants_recon=any(s.metric_id.startswith("reconstruction.") for s in specs)
            core_data={};n_core=0
            act_path=None;code_path=None
            if wants_core or wants_m2:
                with ActivationWriter(cache_root,act_id,width=int(sae.cfg.d_in)) as writer:
                    collect_activations(model,records,writer,site=target.site)
                    act_path=writer.path
                code_id=CodeIdentity(sae_key,sae_sha,int(sae.cfg.d_in),int(sae.cfg.d_sae))
                code_path,_=encode_from_store(act_path,sae,code_id)
            if wants_core:
                core_data,n_core=calculate_core(act_path,code_path,sae,
                                             sample_ids=[s.record.sample_id for s in config.samples])
            for spec in specs:
                if spec.metric_id.startswith("core."):
                    value=core_data.get(spec.metric_id)
                    results.append(_metric(config,target,sae_key,spec,
                        "completed" if value is not None else "undefined",
                        value,n_valid=n_core if value is not None else 0,
                        reason=None if value is not None else "no valid M1 rows"))
            if wants_m2:
                meas=root/"measurements"/act_id.key/sae_sha
                with ResultWriter(meas,identity,resume=resume) as writer:
                    BatchEngine(model,sae,identity,activation_reader=ShardedReader(act_path),
                                code_reader=ShardedReader(code_path)).run(
                        config.samples,config.conditions,writer,
                        sample_batch=config.sample_batch,microbatch=config.microbatch,
                        budget=RunBudget(max_measurements=config.max_measurements),
                        tolerance=config.tolerance)
                measurements,raw=calculate_interventions(meas,identity,conditions=config.conditions)
                for spec in specs:
                    if not spec.metric_id.startswith("intervention."):
                        continue
                    for c in config.conditions:
                        item=measurements.get((c.name,spec.metric_id))
                        if item:
                            value,n,refs=item
                            results.append(_metric(config,target,sae_key,spec,"completed",value,
                                                   n_valid=n,condition=c.name,refs=refs))
                        else:
                            results.append(_metric(config,target,sae_key,spec,"undefined",
                                reason="metric not recorded for this condition",condition=c.name))
            if wants_recon:
                observations=[]
                for s in config.samples:
                    if s.target_token_id is not None:
                        observations.append(target_nll(model,sae,s,target.site))
                for spec in specs:
                    if not spec.metric_id.startswith("reconstruction."):
                        continue
                    if spec.metric_id=="reconstruction.target_nll":
                        val=sum(patched for _,patched in observations)/len(observations) if observations else None
                        results.append(_metric(config,target,sae_key,spec,
                            "completed" if val is not None else "undefined",val,
                            n_valid=len(observations) if val is not None else 0,
                            reason=None if val is not None else "no target tokens"))
                    else:
                        ablation=config.options.get("ablation_reference_nll")
                        if type(ablation) not in (float,int) or not observations:
                            results.append(_metric(config,target,sae_key,spec,"undefined",
                                    reason="loss recovered requires explicit ablation_reference_nll and target tokens"))
                        else:
                            base=sum(v[0] for v in observations)/len(observations)
                            patch=sum(v[1] for v in observations)/len(observations)
                            val=loss_recovered(base,patch,float(ablation))
                            results.append(_metric(config,target,sae_key,spec,
                                "completed" if val is not None else "undefined",val,
                                n_valid=len(observations) if val is not None else 0,
                                reason=None if val is not None else "ablation and baseline losses coincide"))
            for paper in ("sar","rss"):
                subset=[spec for spec in specs if spec.metric_id.startswith(paper+".")]
                if not subset:
                    continue
                source=config.sources.get(paper)
                if not source:
                    for spec in subset:
                        results.append(_metric(config,target,sae_key,spec,"blocked",
                                               reason=f"missing sources.{paper} original estimator"))
                    continue
                try:
                    vals,n,digest=run_paper([s.metric_id for s in subset],model,sae,target.site,
                            source=source,options=config.options,samples=config.samples,seed=config.seed)
                    for spec in subset:
                        item=vals.get(spec.metric_id.removeprefix(paper+"."))
                        results.append(_metric(config,target,sae_key,spec,
                            "completed" if item is not None and math.isfinite(item) else "undefined",
                            float(item) if item is not None and math.isfinite(item) else None,
                            n_valid=n if item is not None and math.isfinite(item) else 0,
                            reason=None if item is not None and math.isfinite(item) else "upstream estimator undefined",
                            source_sha=digest))
                except Exception as exc:
                    for spec in subset:
                        results.append(_metric(config,target,sae_key,spec,"blocked",
                                               reason=f"{type(exc).__name__}: {exc}"))
            for spec in specs:
                if not spec.metric_id.startswith("saebench."):
                    continue
                source=config.sources.get(spec.metric_id)
                if not source:
                    results.append(_metric(config,target,sae_key,spec,"blocked",
                        reason="native SAEBench output path required; no local upstream run executed"))
                    continue
                try:
                    val,digest,commit=import_native(source,spec.metric_id,
                            expected_commit=config.options.get("saebench_commit"))
                    results.append(_metric(config,target,sae_key,spec,"completed",val,n_valid=1,
                        source_sha=digest,source_commit=commit))
                except Exception as exc:
                    results.append(_metric(config,target,sae_key,spec,"blocked",
                        reason=f"{type(exc).__name__}: {exc}"))
    store_results(root,results,run_id=config.run_id)
    write_report(root,config,plan,results,provenance=provenance)
    return [r.to_dict() for r in results]
