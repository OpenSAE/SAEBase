"""M2 numerical, batching, provenance, resume and steering contracts (CPU)."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter, capture_features, evaluate_edit
from saebase.store import (ActivationIdentity, ActivationWriter, CodeIdentity, SampleRecord,
                           ShardedReader, TokenSelection, encode_from_store, sha256_state_dict)
from saebase.interventions import (BatchEngine, Condition, ExperimentIdentity,
                                   GenerationSpec, MeasurementIntegrityError,
                                   MeasurementSample, ResultWriter, RunBudget, apply_condition,
                                   compare_steered_generation, dry_run, read_results, run_steering)


def context(*, backend_model=None):
    m = ModelAdapter((backend_model or ResearchToyModel()).eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    ident = ExperimentIdentity(sha256_state_dict(m.model), sha256_state_dict(sae), site,
                               model_key="toy", sae_key="toy", backend=m.backend)
    return m, sae, ident


def records():
    return [MeasurementSample(SampleRecord(
        f"sample-{i}", tuple(range(1, n+1)), selection=TokenSelection("last_valid"),
        prompt_group_id=f"g-{i}", split="test"), target_token_id=1, reference_token_id=0)
        for i,n in enumerate((3,5,4))]


def controls():
    return [Condition("identity", "identity"), Condition("ablate0", "ablate", (0,)),
            Condition("scale", "scale", (0,1), 0.5), Condition("clamp", "clamp", (0,), 1.0),
            Condition("add", "add", (0,2), 0.1),
            Condition("coalition", "coalition", (0,1), retained_ids=(1,)),
            Condition("reconstruction", "reconstruction")]


def test_all_operators_match_single_reference(tmp_path):
    m, sae, ident = context()
    samples = records()
    with ResultWriter(tmp_path, ident, shard_rows=3) as w:
        stats = BatchEngine(m, sae, ident, pad_token_id=0).run(samples, controls(), w,
                          sample_batch=3, microbatch=4)
    assert stats.measured == len(samples)*len(controls())
    assert stats.baseline_batches == 1 and stats.patch_batches < stats.measured
    results = read_results(tmp_path, ident)
    assert len(results) == len(samples)*len(controls())
    for sample in samples:
        target = {r["condition_name"]: r for r in results if r["sample_id"] == sample.record.sample_id}
        for condition in controls():
            r = target[condition.name]
            snapshot = capture_features(m, sae, site=ActivationSite(0, ident.site.hook_name, sample.patch_position),
                                        tokens=m.tokens(tokens=torch.tensor([sample.tokens])))
            patch = apply_condition(sae, snapshot, condition)
            base = m.forward_logits(tokens=snapshot.tokens)
            expected = m.patch_logits(tokens=snapshot.tokens, site=snapshot.site, activation=patch.activation)
            assert r["metrics"]["logit_delta_l2"] == pytest.approx(float(torch.linalg.vector_norm(expected-base)), abs=1e-5)
            assert r["metrics"]["target_logit_delta"] == pytest.approx(float(expected[1]-base[1]),abs=1e-5)
            assert r["metrics"]["kl_baseline_to_edited"] >= -1e-6
        assert target["identity"]["metrics"]["logit_delta_l2"] == pytest.approx(0,abs=1e-6)
        assert target["ablate0"]["metrics"]["activation_delta_l2"] > 0
    assert stats.interventions_per_second > 0


def test_identity_and_reconstruction_are_distinct():
    m, sae, ident = context()
    snapshot = capture_features(m, sae, site=ident.site, prompt="2 3 4")
    ident_patch = apply_condition(sae, snapshot, Condition("i", "identity"))
    full_patch = apply_condition(sae, snapshot, Condition("r", "reconstruction"))
    torch.testing.assert_close(ident_patch.activation, snapshot.activation)
    torch.testing.assert_close(full_patch.activation, snapshot.reconstruction)
    assert not torch.allclose(full_patch.activation, snapshot.activation)


@pytest.mark.parametrize("bad",[
    lambda: Condition("", "identity"),
    lambda: Condition("x", "wrong", (0,)),
    lambda: Condition("x", "scale", (0,), float("nan")),
    lambda: Condition("x", "clamp", (0,0), 2.0),
    lambda: Condition("x", "coalition", (0,), retained_ids=(1,)),
    lambda: Condition("x", "identity", (1,)),
    lambda: Condition("x", "ablate", (0,), 1.0),
    lambda: Condition("x", "add", (True,), 3.0),
])
def test_invalid_condition_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        bad()


def test_invalid_measurement_and_dry_run_budget():
    with pytest.raises(ValueError,match="one patch position"):
        MeasurementSample(SampleRecord("a", (1,2,3), selection=TokenSelection("all_valid")))
    with pytest.raises(ValueError,match="right-padded"):
        MeasurementSample(SampleRecord("a",(1,2,3),attention_mask=(0,1,1)))
    s=records()
    with pytest.raises(ValueError,match="budget"):
        dry_run(s,controls(),budget=RunBudget(max_measurements=20))
    with pytest.raises(ValueError,match="duplicate sample"):
        dry_run([s[0],s[0]], controls())
    with pytest.raises(ValueError,match="duplicate condition"):
        dry_run(s,[controls()[0], controls()[0]])
    assert dry_run(s, controls())["estimated_runtime_seconds"] is None


def test_resume_and_manifest_corruption_rejection(tmp_path):
    m, sae, ident = context()
    samples=records()
    eng=BatchEngine(m,sae,ident,pad_token_id=0)
    with ResultWriter(tmp_path,ident,shard_rows=2) as w:
        first=eng.run(samples,controls(),w,microbatch=3)
        second=eng.run(samples,controls(),w,microbatch=3)
    assert first.measured == 21
    assert second.measured == 0 and second.reused == 21
    assert second.baseline_batches == 0 and second.patch_batches == 0
    data=read_results(tmp_path,ident)
    assert len({r['measurement_id'] for r in data})==21
    with pytest.raises(MeasurementIntegrityError):
        with ResultWriter(tmp_path,replace(ident,model_key="another")):
            pass
    shard = tmp_path / "shard-000000.jsonl"
    shard.write_bytes(shard.read_bytes()+b"corrupt")
    with pytest.raises(MeasurementIntegrityError, match="corrupted"):
        with ResultWriter(tmp_path,ident):
            pass


def test_uncommitted_orphan_is_not_visible_and_can_be_repaired(tmp_path):
    m, sae, ident = context()
    with ResultWriter(tmp_path,ident,shard_rows=100) as w:
        BatchEngine(m,sae,ident).run(records()[:1],controls()[:1],w)
    (tmp_path/"shard-000001.jsonl").write_text("uncommitted junk")
    with ResultWriter(tmp_path,ident) as w:
        BatchEngine(m,sae,ident).run(records()[1:2],controls()[:1],w)
    assert len(read_results(tmp_path,ident))==2


def test_m1_cache_reuse_without_model_activation_capture(tmp_path):
    m, sae, ident = context()
    sample=records()[0]
    h=capture_features(m,sae,site=ident.site,tokens=m.tokens(tokens=torch.tensor([sample.tokens])))
    aident=ActivationIdentity("toy", ident.model_state_sha256,"b"*64,ident.site.hook_name,0)
    with ActivationWriter(tmp_path/"activations",aident,width=4) as w:
        w.append(sample.record, h.activation.unsqueeze(0));path=w.path
    cid=CodeIdentity("toy",ident.sae_state_sha256,4,7)
    code_path,_=encode_from_store(path,sae,cid)
    act=ShardedReader(path);codes=ShardedReader(code_path)
    eng=BatchEngine(m,sae,ident,activation_reader=act,code_reader=codes)
    original=m.model.run_with_cache
    # Forward and patch still run: only activation-capture cache calls prohibited.
    def guarded(ids, names_filter=None):
        raise AssertionError("cache hit should not call run_with_cache")
    m.model.run_with_cache=guarded
    try:
        with ResultWriter(tmp_path/"results",ident) as w:
            stats=eng.run([sample],controls()[:3],w)
        assert stats.measured==3
    finally:
        m.model.run_with_cache=original
    other=replace(ident,sae_state_sha256="e"*64)
    with pytest.raises(ValueError,match="SAE state"):
        BatchEngine(m,sae,other,activation_reader=act,code_reader=codes)


def test_independent_steering_randomness_and_resume(tmp_path):
    m,sae,ident=context()
    samples=records()[:2]
    condition=Condition("strong", "scale", (0,), 0.0)
    spec=GenerationSpec(max_new_tokens=4,seed=17)
    a=compare_steered_generation(m,sae,site=ident.site,sample=samples[0],condition=condition,spec=spec)
    b=compare_steered_generation(m,sae,site=ident.site,sample=samples[0],condition=condition,spec=spec)
    assert a==b and len(a['baseline_generated_token_ids'])==4
    with ResultWriter(tmp_path,ident) as w:
        assert run_steering(m,sae,ident,samples,condition,spec,w)['measured']==2
        assert run_steering(m,sae,ident,samples,condition,spec,w)['reused']==2
    assert len(read_results(tmp_path,ident))==2
    with pytest.raises(ValueError,match="budget"):
        with ResultWriter(tmp_path,ident) as w:
            run_steering(m,sae,ident,samples,condition,spec,w,budget=RunBudget(max_generated_tokens=10))


def test_identity_steering_equals_baseline_and_categorical_seed():
    m,sae,ident=context()
    sample=records()[0]
    for temperature in (0., 1.0):
        spec=GenerationSpec(max_new_tokens=3,temperature=temperature,top_k=2,seed=2026)
        result=compare_steered_generation(m,sae,site=ident.site,sample=sample,
                                          condition=Condition('nothing','identity'),spec=spec)
        assert result['same_continuation'] is True
        assert result['first_different_generated_index'] is None


def test_mixed_token_lengths_require_padding_and_batched_paths(tmp_path):
    m,sae,ident=context()
    with ResultWriter(tmp_path,ident) as w:
        with pytest.raises(ValueError,match="pad_token_id"):
            BatchEngine(m,sae,ident).run(records(),controls()[:1],w)
    with ResultWriter(tmp_path,ident) as w:
        stats=BatchEngine(m,sae,ident,pad_token_id=0).run(records(),controls()[:1],w)
    assert stats.measured==3


def test_wrong_checkpoint_rejected(tmp_path):
    m,sae,ident=context()
    with torch.no_grad():
        next(sae.parameters()).add_(1)
    with pytest.raises(ValueError,match="SAE state"):
        BatchEngine(m,sae,ident)


def test_steering_rejects_invalid_generation_configuration():
    for kwargs in ({'max_new_tokens':0},{'temperature':-1},{'top_k':0},{'seed':-1},
                   {'eos_token_ids':(1,1)},{'max_context_tokens':0}):
        with pytest.raises(ValueError):
            GenerationSpec(**kwargs)


class ToyHFHooks(ResearchToyModel):
    """Wrap the same toy computation in the real HFHooks adapter contract."""
    _saebase_backend = "hf_hooks"

    def get_resid(self, input_ids, layer, offsets):
        assert layer == 0
        h = self._hidden(input_ids)
        return torch.stack([h[0, h.shape[1]-1-int(off)] for off in offsets])

    def run_with_patch(self, input_ids, layer, patches, *, readout_offset=0):
        assert layer == 0
        h = self._hidden(input_ids).clone()
        for off, vec in patches.items():
            h[0, h.shape[1]-1-int(off)] = vec.to(h.device)
        return self._logits(h)[0, h.shape[1]-1-int(readout_offset)]


def test_hf_hooks_fallback_is_explicit_and_matches_sequential(tmp_path):
    m, sae, ident = context(backend_model=ToyHFHooks())
    samples=records()
    conditions=controls()
    with ResultWriter(tmp_path,ident) as w:
        stats=BatchEngine(m,sae,ident).run(samples,conditions,w,sample_batch=3,microbatch=4)
    assert stats.backend=="hf_hooks" and stats.measured==21
    # Single-example forwards are reported as such; no fake batch throughput.
    assert stats.baseline_batches==3 and stats.patch_batches>=21
    rows=read_results(tmp_path,ident)
    r=next(x for x in rows if x['sample_id']==samples[0].record.sample_id and x['condition_name']=="ablate0")
    original=evaluate_edit(m,sae,site=ident.site,tokens=m.tokens(tokens=torch.tensor([samples[0].tokens])),
                           replacements={0:0.0})
    assert r['metrics']['logit_delta_l2']==pytest.approx(float(torch.linalg.vector_norm(original.logit_delta)),abs=1e-5)


def test_oom_patch_microbatch_retries_without_changing_results(tmp_path, monkeypatch):
    m,sae,ident=context()
    eng=BatchEngine(m,sae,ident,pad_token_id=0)
    original=eng._patched_logits
    def out_of_memory_for_large(jobs):
        if len(jobs)>2:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return original(jobs)
    monkeypatch.setattr(eng,'_patched_logits',out_of_memory_for_large)
    with ResultWriter(tmp_path,ident) as w:
        stat=eng.run(records()[:2],controls()[:4],w,microbatch=8,sample_batch=2,
                     verify_identity=False)
    assert stat.microbatch_retries>0 and stat.peak_microbatch<=2
    assert stat.measured==8 and len(read_results(tmp_path,ident))==8


def test_strength_scan_has_distinct_measurement_ids_and_no_aliasing(tmp_path):
    m,sae,ident=context()
    conditions=[Condition(f'scale_{strength}', 'scale', (0,),strength) for strength in (0.0,0.5,1.0,2.0)]
    with ResultWriter(tmp_path,ident) as w:
        BatchEngine(m,sae,ident).run(records()[:1],conditions,w)
    rows=read_results(tmp_path,ident)
    assert len({x['measurement_id'] for x in rows})==4
    assert next(x for x in rows if x['condition_name']=='scale_1.0')['metrics']['activation_delta_l2']==pytest.approx(0,abs=1e-6)
    assert next(x for x in rows if x['condition_name']=='scale_0.0')['metrics']['activation_delta_l2']>0


def test_refuse_relabelled_code_cache_source(tmp_path):
    m,sae,ident=context()
    sample=records()[0]
    sn=capture_features(m,sae,site=ident.site,tokens=m.tokens(tokens=torch.tensor([sample.tokens])))
    aid=ActivationIdentity('toy',ident.model_state_sha256,'b'*64,ident.site.hook_name,0)
    with ActivationWriter(tmp_path/'cache',aid,width=4) as w:
        w.append(sample.record,sn.activation[None]);apath=w.path
    # Other SAE code cache must not be treated as a match for this experiment.
    alternate=ResearchToySAE().eval()
    with torch.no_grad():
        alternate.bias.add_(.2)
    cid=CodeIdentity('other',sha256_state_dict(alternate),4,7)
    cp,_=encode_from_store(apath,alternate,cid)
    with pytest.raises(ValueError,match='SAE checkpoint differs'):
        BatchEngine(m,sae,ident,activation_reader=ShardedReader(apath),code_reader=ShardedReader(cp))


def test_configurable_cli_dry_run_requires_no_remote_weights(tmp_path):
    import subprocess,sys
    from pathlib import Path
    conf=Path(__file__).resolve().parents[1]/'conf/interventions/m2_pythia_example.json'
    repo=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,'scripts/m2_batch.py','--config',str(conf),'--dry-run'],
                     cwd=repo,capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
    obj=json.loads(r.stdout)
    assert obj['measurements']==4 and obj['steering_total_max_generated_tokens']==6
    assert not (repo/'experiment/m2').exists()
    r=subprocess.run([sys.executable,'scripts/m2_batch.py','--config',str(conf)],
                     cwd=repo,capture_output=True,text=True,timeout=20)
    assert r.returncode!=0 and '--real' in r.stderr
