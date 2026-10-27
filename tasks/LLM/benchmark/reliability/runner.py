"""Audit frozen M3 outputs; never calls the model or modifies M3 metrics."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from itertools import combinations
from dataclasses import replace

from tasks.LLM.benchmark.backend.registry import get_metric
from .schema import AuditConfig
from .observations import _read_m3, observations_for_metric
from .bootstrap import bootstrap_metric,paired_bootstrap,sample_size_curve
from .ranking import rank_stability
from .localization import audit_localization,compare_localization

def evaluate_m3(root:str|Path,cfg:AuditConfig,*,sae_resolver=None,localization_paper=None)->dict:
    root=Path(root)
    config,metrics,provenance=_read_m3(root,localization_paper=localization_paper)
    if cfg.metric_ids:
        unknown=set(cfg.metric_ids)-set(x["metric_id"] for x in metrics)
        if unknown:
            raise ValueError(f"M4 audit requests metrics absent from M3: {sorted(unknown)}")
    entries=[]
    localization_entries=[]
    localization_items=[]
    sae_cache={}
    population=defaultdict(dict)
    all_by_metric=defaultdict(list)
    for m in metrics:
        if cfg.metric_ids and m["metric_id"] not in cfg.metric_ids:
            continue
        if m["metric_id"].startswith("localization."):
            if m["status"] != "completed":
                localization_entries.append({
                    "model_id":m["model_id"],"sae_id":m["sae_id"],
                    "metric_id":m["metric_id"],"status":"blocked",
                    "reason":f"M3 metric status is {m['status']}",
                })
                continue
            audit, artifact = audit_localization(
                root,config,m,cfg,paper=localization_paper
            )
            localization_entries.append(audit)
            localization_items.append((m,artifact,audit))
            continue
        key=(m["model_id"],m["dataset_id"],m["split"],m["metric_id"],m["condition_id"])
        sae=None
        if m["metric_id"] in {"core.reconstruction_mse","core.fvu","core.explained_variance"} and sae_resolver:
            pair=(m["model_id"],m["sae_id"],m["site"]["layer"])
            if pair not in sae_cache:
                sae_cache[pair]=sae_resolver(*pair)
            sae=sae_cache[pair]
        rows,reason=observations_for_metric(root,config,m,provenance,sae=sae)
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
    primary={tuple(sorted(x)) for x in cfg.primary_comparisons}
    for (model,dataset,split,metric,condition),saes in sorted(population.items(),key=lambda kv:str(kv[0])):
        candidates=[(a,b) for a,b in combinations(sorted(saes),2)
                    if not primary or (a,b) in primary]
        for left,right in candidates:
            # Apply the family-wise Bonferroni adjustment to explicitly
            # predeclared primary comparisons only; exploratory results
            # retain unadjusted descriptive 95% intervals.
            n_primary=len(primary) if primary else 0
            adjusted=replace(cfg,confidence=1-(1-cfg.confidence)/n_primary) if n_primary>1 else cfg
            result=paired_bootstrap(saes[left],saes[right],adjusted)
            comparisons.append({"model_id":model,"dataset_id":dataset,"split":split,
                                "metric_id":metric,"condition_id":condition,
                                "left_sae":left,"right_sae":right,
                                "comparison_type":"primary" if primary else "exploratory",
                                "multiple_comparison_method":"bonferroni" if n_primary>1 else "none",
                                **result})
        if len(saes)>1:
            rankings.append(rank_stability(saes,cfg))
    localize_comparisons=compare_localization(localization_items)
    return {"schema":"saebase.m4.audit.v1","status":"completed",
            "m3_run_id":config.run_id,"audit_config":cfg.to_dict(),
            "resampling_unit":"prompt_group_id","ci_method":"percentile",
            "comparison_policy":"family-wise Bonferroni for declared primary; exploratory unadjusted",
            "metric_audits":entries,"paired_comparisons":comparisons,
            "localization_audits":localization_entries,
            "localization_comparisons":localize_comparisons,
            "rank_stability":rankings,
            "unsupported_metrics":sum(x["status"]=="unsupported" for x in entries),
            "undefined_metrics":sum(x["status"]=="undefined" for x in entries+localization_entries),
            "model_forwards_during_audit":0}
