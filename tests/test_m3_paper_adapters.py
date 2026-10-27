"""Optional faithful numerical comparisons with the original paper modules."""
import os
from pathlib import Path
import pytest
from saebase.saebase.examples._synthetic_common import ResearchToyModel,ResearchToySAE
from saebase.api import ModelAdapter,ActivationSite
from saebase.store import SampleRecord
from saebase.interventions import MeasurementSample
from saebase.benchmarks.adapters.papers import run_paper

SITE=ActivationSite(0,"blocks.0.hook_resid_post")

def _source(name):
    value=os.getenv("SAEBASE_"+name.upper()+"_SOURCE")
    if not value:
        pytest.skip("original "+name+" repo not installed")
    return Path(value)

def test_m3_calls_original_sar_unmodified():
    model=ModelAdapter(ResearchToyModel().eval())
    sae=ResearchToySAE().eval()
    sample=MeasurementSample(SampleRecord("s1",(2,3,4),split="test"),
                             target_token_id=0,reference_token_id=1)
    values,n,sha=run_paper(["sar.sar1","sar.sar2","sar.rank2"],
        model,sae,SITE,source=_source("sar"),
        options={"sar":{"K":3,"M":80,"n_fit":55}},samples=[sample],seed=3)
    assert n==80
    assert len(sha)==64
    assert values["sar1"]==pytest.approx(0.2156495440166672,abs=1e-6)
    assert values["sar2"]<1e-8

def test_m3_calls_original_rss_unmodified():
    model=ModelAdapter(ResearchToyModel().eval())
    sae=ResearchToySAE().eval()
    raw=[("clean","clean","1 2 3")]+[
       (f"selection{i}","selection",f"1 2 {i}") for i in range(4,9)]+[
       (f"held{i}","held_out",f"1 2 {i}") for i in range(9,14)]
    samples=[MeasurementSample(SampleRecord(i,tuple(map(int,prompt.split())),variant_role=role,
                split="selection" if role=="selection" else "held_out" if role=="held_out" else "test"))
             for i,role,prompt in raw]
    opts={"rss":{"prompts":{i:p for i,_,p in raw},"M":20,"K":5}}
    values,n,sha=run_paper(["rss.mean_rss","rss.held_out_persistence"],
            model,sae,SITE,source=_source("rss"),options=opts,samples=samples,seed=0)
    assert n==5
    assert len(sha)==64
    assert values["mean_rss"]==pytest.approx(0.513370703458786,rel=1e-5)
    assert values["held_out_persistence"]==pytest.approx(1.0,abs=1e-6)
