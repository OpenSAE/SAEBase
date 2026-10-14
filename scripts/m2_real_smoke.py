#!/usr/bin/env python3
"""Real M2 batch + M1-cache + M0-reference smoke (Pythia 70M, CPU).

Downloads public checkpoints on demand. Success evidence is never created from
fake model weights or from a synthetic fixture alone.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from saebase.api import ActivationSite, capture_features, edit_features, evaluate_edit, load_registered
from saebase.store import (ActivationIdentity, ActivationWriter, CodeIdentity,
                           SampleRecord, ShardedReader, TokenSelection, collect_activations,
                           encode_from_store, fingerprint, sha256_state_dict)
from saebase.interventions import (BatchEngine, Condition, ExperimentIdentity,
                                   GenerationSpec, MeasurementSample, ResultWriter,
                                   RunBudget, dry_run, read_results, run_steering)



def compare_m0_activation_probe(reference, observed, *, atol: float, rtol: float) -> float:
    """Check the stored M0 probe coordinatewise using its recorded tolerances.

    M0 uses numpy-style |observed-reference| <= atol + rtol*|reference|.
    The returned max *absolute* difference is a diagnostic, not the acceptance
    criterion. Reject changed probe geometry or non-finite values explicitly.
    """
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("M0 numerical tolerances must be finite and nonnegative")
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(observed, dtype=np.float64)
    if expected.ndim != 1 or expected.size == 0 or expected.shape != actual.shape:
        raise RuntimeError("M0 activation probe dimensions differ")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError("M0 activation probe contains non-finite values")
    delta = np.abs(actual - expected)
    allowed = atol + rtol * np.abs(expected)
    if np.any(delta > allowed):
        pos = int(np.argmax(delta > allowed))
        raise RuntimeError(
            f"M0 activation probe numerical drift at coordinate {pos}: "
            f"abs_error={delta[pos]:.9g}, allowed={allowed[pos]:.9g}, "
            f"reference={expected[pos]:.9g}"
        )
    return float(delta.max())


def run(reference: Path, cache_root: Path, output: Path) -> dict:
    m0 = json.loads(reference.read_text())
    if m0.get("real_checkpoint_validated") is not True or m0.get("status") != "passed":
        raise RuntimeError("M0 fixture must be an actual validated checkpoint run")
    torch.manual_seed(0)
    bundle = load_registered(model_key="pythia70m",sae_key="pythia70m_resid_sm",layer=3,device="cpu")
    model,sae = bundle.model,bundle.sae
    model_sha,sae_sha = sha256_state_dict(model.model),sha256_state_dict(sae)
    assert model_sha == m0["model_state_sha256"]
    assert sae_sha == m0["sae_state_sha256"]
    site=ActivationSite(3,"blocks.3.hook_resid_post")
    sample_rows = m0["samples"][:3]
    if len(sample_rows) != 3:
        raise RuntimeError("three real M0 reference prompts are required")
    samples = [MeasurementSample(SampleRecord(
        sample_id=row["sample_id"],token_ids=tuple(row["token_ids"]),
        selection=TokenSelection("last_valid"),split="m0-reference",
        prompt_group_id=row["sample_id"]),target_token_id=1,reference_token_id=0)
        for row in sample_rows]
    # Validate exact public-reference probes before trusting cached interventions.
    reference_tolerance = m0.get("tolerances")
    if not isinstance(reference_tolerance, dict) or not all(
        name in reference_tolerance for name in ("atol", "rtol")
    ):
        raise ValueError("M0 reference is missing its recorded numerical tolerances")
    errors = []
    for sample,old in zip(samples,sample_rows):
        snap = capture_features(model,sae,site=site,
                tokens=model.tokens(tokens=torch.tensor([sample.tokens])))
        errors.append(compare_m0_activation_probe(
            old["activation_probe"],
            snap.activation.detach().cpu().numpy()[:len(old["activation_probe"])],
            atol=float(reference_tolerance["atol"]),
            rtol=float(reference_tolerance["rtol"]),
        ))
        if int(torch.count_nonzero(snap.codes)) != old["codes_l0"]:
            raise RuntimeError("M0 SAE code sparsity mismatch")
    tokenizer=model.model.tokenizer
    token_digest=fingerprint({"vocab":tokenizer.get_vocab(),
                              "special_tokens":getattr(tokenizer,"special_tokens_map",{}),
                              "chat_template":getattr(tokenizer,"chat_template",None)})
    act_id=ActivationIdentity("pythia70m",model_sha,token_digest,site.hook_name,3,
                              execution={"model_backend":model.backend,"run_mode":"eval","device":"cpu"},
                              preprocessing={"source":"M0 SimpleCopying token IDs","selection":"last_valid"})
    with ActivationWriter(cache_root,act_id,width=512,shard_target_bytes=4096) as writer:
        collected=collect_activations(model,[x.record for x in samples],writer,site=site)
        apath=writer.path
    assert collected.collected == len(samples)
    cident=CodeIdentity("pythia70m_resid_sm",sae_sha,512,32768)
    cpath, encoded=encode_from_store(apath,sae,cident,layout="csr",max_rows=16,
                                     shard_target_bytes=256*1024)
    first_feature=int(sample_rows[0]["selected_feature"])
    another_feature=(first_feature+1) % 32768
    conditions=[Condition("identity","identity"),
                Condition("ablate","ablate",(first_feature,)),
                Condition("half_scale","scale",(first_feature,),0.5),
                Condition("clamp_zero","clamp",(first_feature,),0),
                Condition("add","add",(first_feature,),0.2),
                Condition("coalition","coalition",(first_feature,another_feature),retained_ids=(another_feature,)),
                Condition("reconstruction","reconstruction")]
    identity=ExperimentIdentity(model_sha,sae_sha,site,
            model_key="pythia70m",sae_key="pythia70m_resid_sm",tokenizer_sha256=token_digest,
            backend=model.backend)
    engine=BatchEngine(model,sae,identity,activation_reader=ShardedReader(apath),
                       code_reader=ShardedReader(cpath),pad_token_id=0)
    experiment=output.parent/"measurements"
    with ResultWriter(experiment,identity,shard_rows=6) as writer:
        stats=engine.run(samples,conditions,writer,sample_batch=3,microbatch=7,
                         tolerance=1e-3)
        # A repeated run cannot perform any forward or patch when fully committed.
        original_forward=model.model.forward
        original_hooks=model.model.run_with_hooks
        def disallow(*_a,**_k):
            raise AssertionError("fully resumed M2 experiment made an extra model forward")
        model.model.forward=disallow
        model.model.run_with_hooks=disallow
        try:
            resumed=engine.run(samples,conditions,writer,sample_batch=3,microbatch=7,
                               tolerance=1e-3)
        finally:
            model.model.forward=original_forward
            model.model.run_with_hooks=original_hooks
    if stats.measured != 21 or resumed.reused != 21 or resumed.measured != 0:
        raise RuntimeError("M2 measurements or resume incomplete")
    results=read_results(experiment,identity)
    assert len(results)==21
    zero=[r for r in results if r["condition_name"]=="identity"]
    identity_max_error = max(r["metrics"]["logit_delta_max_abs"] for r in zero)
    if identity_max_error > 1e-3:
        raise RuntimeError(f"identity intervention exceeds 1e-3 max-logit tolerance: {identity_max_error}")
    first=samples[0]
    expected=evaluate_edit(model,sae,site=site,
        replacements={first_feature:0.0},tokens=model.tokens(tokens=torch.tensor([first.tokens])))
    actual=next(r for r in results if r["sample_id"]==first.record.sample_id and r["condition_name"]=="ablate")
    error=abs(actual["metrics"]["logit_delta_l2"] -
              float(torch.linalg.vector_norm(expected.logit_delta.float()).item()))
    if error > 0.05:
        raise RuntimeError(f"M2 batch L2 result differs from sequential reference by {error}")
    with ResultWriter(output.parent/"steering",identity) as writer:
        steering=run_steering(model,sae,identity,samples[:1],
               Condition("ablate_top","ablate",(first_feature,)),
               GenerationSpec(max_new_tokens=2,seed=0),writer,
               budget=RunBudget(max_generated_tokens=10))
    report={"schema":"saebase.m2.real_batch.v1","status":"passed",
            "real_checkpoint_validated":True,"m0_reference_verified":True,
            "model_state_sha256":model_sha,"sae_state_sha256":sae_sha,
            "samples":len(samples),"measurements":len(results),
            "conditions":len(conditions),"batch_stats":asdict(stats),
            "resume_reused":resumed.reused,"resume_model_forward_calls":0,
            "m0_activation_probe_max_error":max(errors),
            "m0_activation_probe_tolerances": {
                "atol": float(reference_tolerance["atol"]),
                "rtol": float(reference_tolerance["rtol"]),
            },
            "single_vs_batch_logit_delta_l2_abs_error":error,
            "identity_conditions_verified":len(zero),
            "identity_max_logit_abs_error":identity_max_error,
            "real_batch_identity_atol":1e-3,
            "single_batch_l2_abs_tolerance":0.05,
            "steering":steering,"stored_input_activation_rows":ShardedReader(apath).row_count,
            "encoded_sae_samples":encoded.collected}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("[M2] REAL Pythia/SAE batched interventions, cache reuse and steering passed")
    print(json.dumps(report,indent=2,allow_nan=False))
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-reference",type=Path,required=True)
    parser.add_argument("--cache-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.m0_reference,args.cache_root,args.output)

if __name__ == "__main__":
    main()
