"""M5 release gate deliberately fails closed for synthetic/mismatched proof."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

import pytest
from saebase.release.gates import audit_inventory,audit_result,audit_distributions
from saebase.release.schema import load_matrix

ROOT=Path(__file__).resolve().parents[3]
CONFIG=load_matrix(ROOT/"conf/release/m5_matrix.json")
SOURCE=json.loads((ROOT/"conf/release/m5_asset_provenance.json").read_text())

def valid_record(real=True):
    models=[]
    idx=0
    for target in CONFIG.targets:
        checkpoints={}
        for key in target.sae_keys:
            idx+=1
            checkpoints[key]={
                "weight_sha256":f"{idx:064x}","sae_card":target.sae_cards[key],
                "metrics":len(CONFIG.metrics)*len(CONFIG.conditions)}
        models.append({"model_key":target.model_key,"family":target.family,
            "n_prompts":len(CONFIG.prompts),"audit_groups":len(CONFIG.prompts),
            "elapsed_seconds":2.,"peak_process_rss_bytes":1024**2,
            "checkpoints":checkpoints})
    return {"schema":"saebase.m5.results.v1","status":"passed",
            "real_checkpoints":real,"matrix_id":CONFIG.matrix_id,
            "release_version":"0.2.0","models":models}

def test_inventory_cards_and_bundling_blocked():
    out=audit_inventory(SOURCE,CONFIG,license_path=ROOT/"LICENSE")
    assert out["status"]=="documented" and out["count"]==6
    bad=deepcopy(SOURCE);bad["bundled_model_or_sae_weights"]=True
    with pytest.raises(ValueError,match="bundling forbidden"):
        audit_inventory(bad,CONFIG,license_path=ROOT/"LICENSE")
    bad=deepcopy(SOURCE);bad["third_party_assets"][0]["source"]="https://example.invalid/changed"
    with pytest.raises(ValueError,match="provenance differs"):
        audit_inventory(bad,CONFIG,license_path=ROOT/"LICENSE")

def test_exact_real_two_family_distinct_checkpoints_required():
    rec=valid_record()
    assert audit_result(rec,CONFIG)["checkpoints"]==4
    with pytest.raises(ValueError,match="synthetic-only"):
        audit_result(valid_record(False),CONFIG)
    bad=deepcopy(rec);bad["models"][1]["checkpoints"].pop(next(iter(bad["models"][1]["checkpoints"])))
    with pytest.raises(ValueError,match="coverage incomplete"):
        audit_result(bad,CONFIG)
    bad=deepcopy(rec)
    duplicate=next(iter(bad["models"][0]["checkpoints"].values()))["weight_sha256"]
    next(iter(bad["models"][1]["checkpoints"].values()))["weight_sha256"]=duplicate
    with pytest.raises(ValueError,match="distinct checkpoint claims"):
        audit_result(bad,CONFIG)

def test_no_distribution_artifacts_is_failure(tmp_path):
    with pytest.raises(ValueError,match="one SAEBase wheel"):
        audit_distributions(tmp_path)
