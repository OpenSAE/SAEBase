"""Deterministic DAG and explicit, conservative resource estimates."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

from .schema import BenchmarkConfig
from .registry import get_metric

@dataclass(frozen=True)
class PlanNode:
    job_id: str
    operation: str
    dependencies: tuple[str, ...]
    target: str
    cache_status: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def build_plan(config: BenchmarkConfig) -> dict[str, Any]:
    # No weights or real source code is loaded here. Cache hits cannot be
    # verified without resolving exact model/tokenizer/SAE SHA-256 identities.
    specs = [get_metric(m) for m in config.metrics]
    from tasks.LLM.benchmark.extensions.runtime import validate_external_target
    for target in config.targets:
        validate_external_target(target, config.sources)
    pairs = sum(len(t.sae_keys) for t in config.targets)
    wants_localization=any("localization_codes" in s.required_inputs for s in specs)
    if wants_localization:
        from .adapters.localization import parse_localization_options
        parse_localization_options(config)
    wants_m2=any("m2_measurements" in s.required_inputs for s in specs)
    wants_m1=any(bool({"m1_activations","m1_codes","m2_measurements","localization_codes"} & set(s.required_inputs))
                 for s in specs)
    if wants_m2 and not config.conditions:
        raise ValueError("intervention metrics require explicit conditions")
    n_inter = len(config.samples) * len(config.conditions) * pairs if wants_m2 else 0
    n_rows = (sum(len(s.record.token_ids) if s.record.selection.mode == "all_valid"
                  else (len(s.record.selection.indices) if s.record.selection.mode == "indices" else 1)
                  for s in config.samples) * len(config.targets)) if wants_m1 else 0
    if n_inter > config.max_measurements:
        raise ValueError(f"{n_inter} measurements exceed budget {config.max_measurements}")
    if n_rows > config.max_activation_rows:
        raise ValueError(f"{n_rows} activation rows exceed budget {config.max_activation_rows}")
    nodes: list[PlanNode] = []
    for t in config.targets:
        base=f"{t.model_key}/{t.site.hook_name}"
        act=f"{base}/activations"
        if wants_m1:
            nodes.append(PlanNode(act,"m1_capture",(),base))
        for sae in t.sae_keys:
            prefix=f"{base}/{sae}"
            encode=prefix+"/codes"
            if wants_m1:
                nodes.append(PlanNode(encode,"m1_encode",(act,),prefix))
            if wants_localization:
                nodes.append(PlanNode(prefix+"/localization","original_localize_eval",(encode,),prefix))
            if wants_m2:
                nodes.append(PlanNode(prefix+"/measurements","m2_measure",(encode,),prefix))
            if any("model_reconstruction" in s.required_inputs for s in specs):
                nodes.append(PlanNode(prefix+"/reconstruction","behavioral_reconstruction",(),prefix))
            for paper in ("sar","rss"):
                if any(f"paper_{paper}" in s.required_inputs for s in specs):
                    nodes.append(PlanNode(prefix+f"/paper/{paper}","original_paper_eval",(),prefix))
            for spec in specs:
                req=set(spec.required_inputs)
                deps=[]
                if "localization_codes" in req:
                    deps=[prefix+"/localization"]
                elif req & {"m1_activations","m1_codes"}:
                    deps=[encode]
                elif "m2_measurements" in req:
                    deps=[prefix+"/measurements"]
                elif "model_reconstruction" in req:
                    deps=[prefix+"/reconstruction"]
                elif "paper_sar" in req:
                    deps=[prefix+"/paper/sar"]
                elif "paper_rss" in req:
                    deps=[prefix+"/paper/rss"]
                nodes.append(PlanNode(prefix+"/metric/"+spec.metric_id,"metric",
                                      tuple(deps),prefix))
    metric_jobs = tuple(n.job_id for n in nodes if n.operation == "metric")
    nodes.append(PlanNode(config.run_id + "/report", "report", metric_jobs, config.run_id))
    return {"schema": "saebase.benchmark.plan.v1", "run_id": config.run_id,
            "models": len(config.targets), "sae_evaluations": pairs,
            "sample_count": len(config.samples), "metric_count": len(specs),
            "metric_jobs": pairs * len(specs),
            "planned_measurements": n_inter, "planned_activation_rows": n_rows,
            "budget": {"max_measurements": config.max_measurements,
                       "max_activation_rows": config.max_activation_rows},
            "hardware_runtime_seconds": None, "gpu_memory_bytes": None,
            "nodes": [n.to_dict() for n in nodes],
            "note": "Preflight never downloads weights; cache identity and runtime estimates require a verified execution."}
