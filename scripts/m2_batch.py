#!/usr/bin/env python3
"""SAEBase JSON feature-intervention experiment entrypoint (batch + optional token steering).

Dry-run never loads pretrained weights, imports SAE Lens, or creates artifacts.
Actual execution is opt-in with --real; it may download public checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saebase.api import ActivationSite
from saebase.store import SampleRecord, TokenSelection
from saebase.interventions import (BatchEngine, Condition, ExperimentIdentity,
                                   GenerationSpec, MeasurementSample, ResultWriter,
                                   RunBudget, dry_run, run_steering)


def _config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a JSON object")
    allowed = {"model_key", "sae_key", "layer", "samples", "conditions", "pad_token_id",
               "sample_batch", "microbatch", "max_measurements", "max_generated_tokens",
               "max_seconds", "steering", "tolerance"}
    extra = sorted(set(config) - allowed)
    if extra:
        raise ValueError(f"unknown configuration fields: {extra}")
    for key in ("model_key", "sae_key", "layer", "samples", "conditions"):
        if key not in config:
            raise ValueError(f"missing required experiment field: {key}")
    samples = []
    for row in config["samples"]:
        data = dict(row)
        position = data.pop("patch_position", -1)
        readout = data.pop("readout_position", -1)
        target = data.pop("target_token_id", None)
        reference = data.pop("reference_token_id", None)
        record = SampleRecord(sample_id=data["sample_id"],
                token_ids=tuple(data["token_ids"]),
                selection=TokenSelection("last_valid") if position == -1
                    else TokenSelection("indices",(position,)),
                attention_mask=tuple(data["attention_mask"]) if data.get("attention_mask") is not None else None,
                split=data.get("split", "evaluation"),
                prompt_group_id=data.get("prompt_group_id"))
        samples.append(MeasurementSample(record,readout_position=readout,
                target_token_id=target,reference_token_id=reference))
    conditions=[Condition(name=c["name"],operation=c["operation"],
                feature_ids=tuple(c.get("feature_ids",())),value=c.get("value",0.0),
                retained_ids=tuple(c.get("retained_ids",()))) for c in config["conditions"]]
    budget=RunBudget(max_measurements=config.get("max_measurements",100_000),
                     max_generated_tokens=config.get("max_generated_tokens",10_000),
                     max_seconds=config.get("max_seconds"))
    plan=dry_run(samples,conditions,microbatch=config.get("microbatch",16),budget=budget)
    spec=None
    if "steering" in config:
        raw=config["steering"]
        if not isinstance(raw, dict) or "condition_name" not in raw:
            raise ValueError("steering must specify condition_name")
        candidates=[c for c in conditions if c.name == raw["condition_name"]]
        if len(candidates)!=1:
            raise ValueError("steering condition_name must match exactly one condition")
        spec=GenerationSpec(max_new_tokens=raw.get("max_new_tokens",8),
                  seed=raw.get("seed",0),temperature=raw.get("temperature",0.),
                  top_k=raw.get("top_k"),eos_token_ids=tuple(raw.get("eos_token_ids",[])),
                  max_context_tokens=raw.get("max_context_tokens"))
        if 2 * len(samples) * spec.max_new_tokens > budget.max_generated_tokens:
            raise ValueError("steering exceeds token budget")
        plan["steering_total_max_generated_tokens"] = 2 * len(samples) * spec.max_new_tokens
    return config,samples,conditions,budget,plan,spec


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path,help="Versioned JSON experiment configuration")
    parser.add_argument("--output",type=Path,default=Path("experiment/m2"))
    parser.add_argument("--dry-run",action="store_true",help="Validate cost without loading model or SAE")
    parser.add_argument("--real",action="store_true",help="Explicitly permit loading public model and SAE weights")
    parser.add_argument("--no-resume",action="store_true",help="Fail if a results directory exists")
    args=parser.parse_args(argv)
    config,samples,conditions,budget,plan,generation=_config(args.config)
    if args.dry_run:
        print(json.dumps({"schema":"saebase.m2.dry_run.v1",**plan},indent=2))
        return 0
    if not args.real:
        parser.error("execution loads public weights; provide --real or choose --dry-run")
    from saebase.api import load_registered
    from saebase.store import sha256_state_dict
    model_key,sae_key,layer=config["model_key"],config["sae_key"],config["layer"]
    bundle=load_registered(model_key=model_key,sae_key=sae_key,layer=layer,device="cpu")
    site=bundle.site
    ident=ExperimentIdentity(sha256_state_dict(bundle.model.model),
                             sha256_state_dict(bundle.sae),
                             site,model_key=model_key,sae_key=sae_key,backend=bundle.model.backend,
                             metadata={"source":"scripts/m2_batch.py"})
    engine=BatchEngine(bundle.model,bundle.sae,ident,pad_token_id=config.get("pad_token_id"))
    with ResultWriter(args.output/"measurements",ident,resume=not args.no_resume) as writer:
        measured=engine.run(samples,conditions,writer,sample_batch=config.get("sample_batch",8),
                            microbatch=config.get("microbatch",16),budget=budget,
                            tolerance=config.get("tolerance",1e-4))
    result={"schema":"saebase.m2.report.v1","experiment":ident.to_dict(),
            "status":"passed","measurements":asdict(measured),
            "resolved_plan":plan, "model_weights_loaded": True}
    if generation is not None:
        chosen=next(c for c in conditions if c.name==config["steering"]["condition_name"])
        with ResultWriter(args.output/"steering",ident,resume=not args.no_resume) as writer:
            result["steering"]=run_steering(bundle.model,bundle.sae,ident,samples,chosen,generation,
                                            writer,budget=budget)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
