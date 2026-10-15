"""Reconstruct M4 observation units from verified M3/M1/M2 artifacts only.

No pretrained model loads; no forward passes; original M3 metric rows immutable.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json, math
import numpy as np

from saebase.api import ActivationSite
from saebase.benchmarks import BenchmarkConfig
from saebase.benchmarks.results import read_metrics
from saebase.interventions import ExperimentIdentity,read_results
from saebase.store import ShardedReader,resolve_positions,sha256_state_dict
from .schema import Observation

def _read_config(root:Path):
    config=BenchmarkConfig.from_dict(json.loads((root/"resolved_config.json").read_text()))
    if config.run_id!=root.name:
        raise ValueError("M3 run directory does not match frozen configuration ID")
    return config

def _read_m3(root):
    root=Path(root)
    config=_read_config(root)
    rows=read_metrics(root,config.run_id)
    from saebase.benchmarks.runner import validate_external_sources
    validate_external_sources(config,rows)
    provenance=json.loads((root/"provenance.json").read_text())
    return config,rows,provenance

def _id_from_manifest(manifest):
    d=manifest["identity"]
    return ExperimentIdentity(d["model_state_sha256"],d["sae_state_sha256"],
        ActivationSite(**d["site"]),model_key=d["model_key"],sae_key=d["sae_key"],
        tokenizer_sha256=d.get("tokenizer_sha256"),backend=d["backend"],
        metadata=d.get("metadata",{}))

def _find_m2(root,metric,provenance):
    m=metric
    prefix=f"{m['model_id']}/{m['site']['layer']}/{m['sae_id']}"
    expected=provenance.get(prefix)
    if not expected:
        raise ValueError(f"missing M3 checkpoint provenance for {prefix}")
    source=root.parent/"_cache"/"m2"
    matches=[]
    for candidate in sorted(source.glob("*/*/manifest.json")):
        meta=json.loads(candidate.read_text())
        identity=meta.get("identity",{})
        if (identity.get("model_key")==m["model_id"]
                and identity.get("sae_key")==m["sae_id"]
                and identity.get("model_state_sha256")==expected["model_state_sha256"]
                and identity.get("sae_state_sha256")==expected["sae_state_sha256"]
                and identity.get("tokenizer_sha256")==expected["tokenizer_sha256"]
                and identity.get("site")==m["site"]):
            ident=_id_from_manifest(meta)
            rows=read_results(candidate.parent,ident) # M2 verifies shards and row identities
            by_id={r["measurement_id"]:r for r in rows}
            if all(ref in by_id for ref in m["measurement_refs"]):
                matches.append(by_id)
    if len(matches)!=1:
        raise ValueError(f"expected one verified M2 journal for {m['metric_id']}: found {len(matches)}")
    return [matches[0][ref] for ref in m["measurement_refs"]]

def _find_m1_codes(root,metric,provenance,config):
    m=metric
    prefix=f"{m['model_id']}/{m['site']['layer']}/{m['sae_id']}"
    expected=provenance.get(prefix)
    if not expected:
        raise ValueError("missing M3 model/SAE provenance for M1 code extraction")
    matches=[]
    for path in sorted((root.parent/"_cache"/"m1").glob("*/codes/*/manifest.json")):
        meta=json.loads(path.read_text())
        identity=meta.get("identity",{})
        if identity.get("sae",{}).get("sae_state_sha256")!=expected["sae_state_sha256"]:
            continue
        source_meta=json.loads((path.parent.parent.parent/"manifest.json").read_text())
        model=source_meta.get("identity",{})
        if (model.get("model_state_sha256")!=expected["model_state_sha256"]
                or model.get("tokenizer_sha256")!=expected["tokenizer_sha256"]
                or model.get("hook_name")!=m["site"]["hook_name"]
                or model.get("layer")!=m["site"]["layer"]):
            continue
        reader=ShardedReader(path.parent,verify_shards=True)
        if all(reader.has(s.record.sample_id) for s in config.samples):
            matches.append(reader)
    if len(matches)!=1:
        raise ValueError(f"expected one verified M1 code cache: found {len(matches)}")
    return matches[0]

def observations_for_metric(root,config,row,provenance,*,sae=None):
    """Build exactly the M3 observation set for a supported audited metric."""
    metric=row["metric_id"]
    if row["status"]!="completed":
        return (),f"M3 metric status is {row['status']}"
    if row["split"]=="mixed":
        return (),"mixed dataset splits cannot share an audit population"
    samples={s.record.sample_id:s for s in config.samples}
    out=[]
    if metric.startswith("intervention."):
        if not row.get("condition_id") or not row.get("measurement_refs"):
            return (),"intervention raw source references unavailable"
        raw=_find_m2(Path(root),row,provenance)
        for item in raw:
            sid=item["sample_id"]
            sample=samples.get(sid)
            if sample is None:
                raise ValueError(f"unknown source sample {sid}")
            if (item["measurement_identity"]["sample"]!=sample.to_dict()
                    or item["condition_name"]!=row["condition_id"]):
                raise ValueError("M2 source sample/condition identity disagrees with M3 configuration")
            if sample.record.split!=row["split"] or not sample.record.prompt_group_id:
                raise ValueError("M4 requires split-matched nonempty prompt_group_id")
            name=metric.removeprefix("intervention.")
            value=item["metrics"].get(name)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(f"invalid M2 value for {metric}")
            out.append(Observation(row["model_id"],row["sae_id"],metric,row["dataset_id"],
                row["split"],sample.record.prompt_group_id,sid,float(value),
                condition_id=row["condition_id"],measurement_ref=item["measurement_id"],
                seed=row["seed"]))
    elif metric=="core.l0":
        reader=_find_m1_codes(Path(root),row,provenance,config)
        for sample in config.samples:
            if sample.record.split!=row["split"] or not sample.record.prompt_group_id:
                raise ValueError("M4 requires split-matched nonempty prompt_group_id")
            stored=reader.get(sample.record.sample_id)
            if stored.positions!=resolve_positions(sample.record):
                raise ValueError("M1 stored token positions disagree with M3 config")
            # M1 added source_activation_sha256, intentionally not identical
            # to original SampleRecord; verify every other metadata field.
            actual=stored.record.to_dict()
            expected=sample.record.to_dict()
            for key in expected:
                if key!="source_activation_sha256" and actual[key]!=expected[key]:
                    raise ValueError(f"M1 code sample identity mismatch: {key}")
            for pos,z in zip(stored.positions,np.asarray(stored.values)):
                out.append(Observation(row["model_id"],row["sae_id"],metric,row["dataset_id"],
                    row["split"],sample.record.prompt_group_id,sample.record.sample_id,
                    float(np.count_nonzero(z)),measurement_ref=f"m1:{sample.record.sample_id}:{pos}",
                    seed=row["seed"]))
    elif metric in {"core.reconstruction_mse","core.fvu","core.explained_variance"}:
        if sae is None:
            return (),"SAE-only decoder opt-in required; no model forward is needed"
        import torch
        from saebase.sae.contracts import reconstruction_stats
        reader=_find_m1_codes(Path(root),row,provenance,config)
        activation=ShardedReader(reader.path.parents[1],verify_shards=True)
        prefix=f"{row['model_id']}/{row['site']['layer']}/{row['sae_id']}"
        if sha256_state_dict(sae)!=provenance[prefix]["sae_state_sha256"]:
            raise ValueError("SAE decoder checkpoint differs from the frozen M3 weight identity")
        param=next(sae.parameters())
        for sample in config.samples:
            if sample.record.split!=row["split"] or not sample.record.prompt_group_id:
                raise ValueError("M4 requires split-matched nonempty prompt_group_id")
            source=activation.get(sample.record.sample_id)
            code=reader.get(sample.record.sample_id)
            if source.positions!=code.positions or source.positions!=resolve_positions(sample.record):
                raise ValueError("M1 activation/code token selection mismatch")
            with torch.no_grad():
                z=torch.as_tensor(np.asarray(code.values,dtype=np.float32),
                                  dtype=param.dtype,device=param.device)
                reconstructed=sae.decode(z).detach().float().cpu().numpy()
            original=np.asarray(source.values,dtype=np.float64)
            if reconstructed.shape!=original.shape:
                raise ValueError("M1 decoded SAE reconstruction has wrong shape")
            for pos,h,recon in zip(source.positions,original,reconstructed):
                stat=reconstruction_stats(torch.as_tensor(h,dtype=torch.float64),
                    torch.as_tensor(recon,dtype=torch.float64))
                value={"core.reconstruction_mse":stat["reconstruction_mse"],
                       "core.fvu":1.-stat["fve"],
                       "core.explained_variance":stat["fve"]}[metric]
                out.append(Observation(row["model_id"],row["sae_id"],metric,
                    row["dataset_id"],row["split"],sample.record.prompt_group_id,
                    sample.record.sample_id,float(value),
                    measurement_ref=f"m1:{sample.record.sample_id}:{pos}",
                    seed=row["seed"]))
    else:
        return (),f"no raw group-observation adapter for {metric}; original M3 result unchanged"
    if len(out)!=row["n_valid"]:
        raise ValueError(f"{metric} raw sample count {len(out)} differs from M3 n_valid {row['n_valid']}")
    if not out or abs(sum(x.value*x.unit_count for x in out)/sum(x.unit_count for x in out)-row["value"])>max(1e-5,1e-5*abs(row["value"])):
        raise ValueError(f"{metric} raw estimate does not reproduce M3 summary")
    return tuple(out),None
