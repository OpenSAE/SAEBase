"""Audit frozen M3 outputs; never calls the model or modifies M3 metrics."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from itertools import combinations

from saebase.benchmarks.registry import get_metric
from .schema import AuditConfig
from .observations import _read_m3, observations_for_metric
from .bootstrap import bootstrap_metric,paired_bootstrap,sample_size_curve
from .ranking import rank_stability

def evaluate_m3(root:str|Path,cfg:AuditConfig)->dict:
    root=Path(root)
    config,metrics,provenance=_read_m3(root)
    if cfg.metric_ids:
        unknown=set(cfg.metric_ids)-set(x["metric_id"] for x in metrics)
        if unknown:
            raise ValueError(f"M4 audit requests metrics absent from M3: {sorted(unknown)}")
    entries=[]
    population=defaultdict(dict)
    all_by_metric=defaultdict(list)
    for m in metrics:
        if cfg.metric_ids and m["metric_id"] not in cfg.metric_ids:
            continue
        key=(m["model_id"],m["dataset_id"],m["split"],m["metric_id"],m["condition_id"])
        rows,reason=observations_for_metric(root,config,m,provenance)
        if reason:
            entries.append({"status":"unsupported" if m["status"]=="completed" else "blocked",
                            "model_id":m["model_id"],"sae_id":m["sae_id"],
                            "metric_id":m["metric_id"],"condition_id":m["condition_id"],
                            "reason":reason,"n_valid":0})
            continue
        audit=bootstrap_metric(rows,cfg)
        entries.append({"model_id":m["model_id"],"sae_id":m["sae_id"],
                        "metric_id":m["metric_id"],"condition_id":m["condition_id"],
                        "split":m["split"],"dataset_id":m["dataset_id"],
                        "metric_version":m["metric_version"],
                        "protocol_id":m["protocol_id"],"M3_metric_value":m["value"],
                        "audit":audit,"sample_size_curve":sample_size_curve(rows,cfg),
                        "status":audit["status"],"reason":audit.get("reason")})
        population[key][m["sae_id"]]=list(rows)
    comparisons=[]
    rankings=[]
    for (model,dataset,split,metric,condition),saes in sorted(population.items(),key=lambda kv:str(kv[0])):
        for left,right in combinations(sorted(saes),2):
            result=paired_bootstrap(saes[left],saes[right],cfg)
            comparisons.append({"model_id":model,"dataset_id":dataset,"split":split,
                                "metric_id":metric,"condition_id":condition,
                                "left_sae":left,"right_sae":right,**result})
        if len(saes)>1:
            rankings.append(rank_stability(saes,cfg))
    return {"schema":"saebase.m4.audit.v1","status":"completed",
            "m3_run_id":config.run_id,"audit_config":cfg.to_dict(),
            "resampling_unit":"prompt_group_id","ci_method":"percentile",
            "metric_audits":entries,"paired_comparisons":comparisons,
            "rank_stability":rankings,
            "unsupported_metrics":sum(x["status"]=="unsupported" for x in entries),
            "undefined_metrics":sum(x["status"]=="undefined" for x in entries),
            "model_forwards_during_audit":0}
