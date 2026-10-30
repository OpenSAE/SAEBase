"""Real offline loader smoke probe for a trusted, immutable WP21 asset cache.

Only executes with actual dependencies and cached checkpoints. A successful
probe is environment readiness, NOT an exam pass or verification of all
individual case requirements. Run in an isolated executor with network off.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, get_case, sha256, write_json
from preflight import preflight, verify_assets


def probe(case_id: str, model_snapshot: Path, sae_snapshot: Path | None, assets_lock: Path, device: str) -> dict:
    case = get_case(case_id)
    environment = preflight(case_id, model_snapshot=model_snapshot, sae_snapshot=sae_snapshot)
    if environment["blockers"]:
        return {"status":"ENV_BLOCKED", "case_id":case_id, "blockers":environment["blockers"], "model_executed":False}
    lock = verify_assets(assets_lock)
    # Verify supplied snapshots correspond to the same locked bytes. In the
    # HF case snapshots are symlinks to immutable cache blobs.
    from preflight import freeze_assets
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        inferred = freeze_assets([("model", model_snapshot)] + ([('sae', sae_snapshot)] if sae_snapshot else []), Path(temp) / "lock.json")
    if inferred["sha256"] != lock["sha256"]:
        raise ValueError("selected snapshots do not match supplied immutable asset lock")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import torch
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if case["profile"] == "pythia14m_inference":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(model_snapshot), local_files_only=True, trust_remote_code=False)
        model.eval().to(device)
        if getattr(model.config, "model_type", "") != "gpt_neox" or getattr(model.config, "num_hidden_layers", None) != 6 or getattr(model.config, "hidden_size", None) != 128:
            raise ValueError("local Pythia14m snapshot has unexpected pretrained architecture")
        tokens = tokenizer("WP21 offline checkpoint smoke probe.", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            output = model(tokens, output_hidden_states=True)
        finite = bool(torch.isfinite(output.logits).all())
        if not finite or output.logits.shape[0] != 1:
            raise RuntimeError("Pythia model returned invalid logits")
        observed = {"model_type":model.config.model_type,"num_layers":model.config.num_hidden_layers,
                    "hidden_size":model.config.hidden_size,"logits_shape":list(output.logits.shape),
                    "hidden_states":len(output.hidden_states),"dtype":str(output.logits.dtype)}
    elif case["profile"]=="gemma_training" or case_id=="R09":
        # R17 creates its own SAE; R09 is expressly model-only. In both cases
        # a source SAE is not a prerequisite for the cache/load smoke test.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True, trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(str(model_snapshot),local_files_only=True,trust_remote_code=False)
        model.eval().to(device)
        if getattr(model.config,"model_type","") not in {"gemma2","gemma"}:
            raise ValueError("unexpected Gemma model architecture")
        tokens=tokenizer("WP21 offline checkpoint smoke probe.",return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            logits=model(tokens).logits
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite actual Gemma logits")
        observed={"model_type":model.config.model_type,"logits_shape":list(logits.shape),"dtype":str(logits.dtype),
                  "sae_required_at_probe":False,"note":"training or model-only case; task must independently verify its own loaded SAE or provider"}
    else:
        from saebase.api import load_registered, capture_features
        loaded = load_registered(sae_key="gemma2_2b_pt_l25_canonical", model_key="gemma2_2b_base", layer=25, device=device)
        site = loaded.site
        if site.layer != 25 or site.hook_name != "blocks.25.hook_resid_post":
            raise RuntimeError("unexpected Gemma SAE site")
        prompt = "WP21 offline checkpoint smoke probe."
        with torch.no_grad():
            snap = capture_features(model=loaded.model, sae=loaded.sae, prompt=prompt, site=site)
            loaded.model.assert_identity_patch(tokens=snap.tokens, site=site, rtol=1e-4, atol=1e-4)
        if not all(bool(torch.isfinite(t).all()) for t in [snap.activation, snap.codes, snap.reconstruction]):
            raise RuntimeError("nonfinite real SAE capture")
        observed = {"hook":site.hook_name,"layer":site.layer,"backend":loaded.model.backend,
                    "activation_width":snap.activation.numel(),"sae_features":snap.codes.numel(),
                    "token_count":snap.tokens.numel(),"dtype":str(snap.activation.dtype),
                    "model_module":type(loaded.model.model).__module__,"sae_module":type(loaded.sae).__module__}
    verify_assets(assets_lock)  # Detect changed assets during loading.
    return {"schema":"saebase.wp21.real_probe.v1", "status":"REAL_MODEL_PROBE_OK_NOT_EXAM_PASS",
            "case_id":case_id,"profile":case["profile"],"started_at_utc":started,"device":device,"environment":environment,
            "asset_lock_sha256":lock["sha256"],"observed":observed,"model_executed":True,
            "note":"Offline loading and a small real forward succeeded. Reviewer must separately establish actual loader file provenance, case replay, and numerical acceptance."}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True)
    p.add_argument("--model-snapshot",required=True,type=Path)
    p.add_argument("--sae-snapshot",type=Path)
    p.add_argument("--assets-lock",required=True,type=Path)
    p.add_argument("--device",choices=["cpu", "cuda"],default="cpu")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output.exists():
            raise FileExistsError("probe record is immutable")
        result=probe(a.case,a.model_snapshot,a.sae_snapshot,a.assets_lock,a.device)
        write_json(a.output,result,exclusive=True)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 3 if result["status"]=="ENV_BLOCKED" else 0
    except Exception as exc:
        result={"status":"PROBE_FAILED", "case_id":a.case,"error_type":type(exc).__name__,"error":str(exc),"model_executed_successfully":False}
        if not a.output.exists():
            write_json(a.output,result,exclusive=True)
        print(json.dumps(result,ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
