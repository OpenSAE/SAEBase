#!/usr/bin/env python3
"""Opt-in real Pythia M1 cache acceptance compared to the successful M0 run.

Requires the M0 GitHub Actions artifact plus access to public model/SAE weights.
Never claims real checkpoint success from synthetic tests or missing artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from saebase.api import ActivationSite, load_registered
from saebase.store import (ActivationIdentity, ActivationWriter, CodeIdentity,
                           SampleRecord, TokenSelection, ShardedReader, collect_activations,
                           encode_from_store, fingerprint)
from scripts.m0_reference_smoke import _sha_weights


def run(reference: Path, cache_root: Path, output: Path) -> dict:
    m0 = json.loads(reference.read_text(encoding="utf-8"))
    if not m0.get("real_checkpoint_validated") or m0.get("status") != "passed":
        raise ValueError("M0 input must be a successful real-checkpoint reference")
    bundle = load_registered(model_key="pythia70m", sae_key="pythia70m_resid_sm",
                             layer=3, device="cpu")
    model_sha = _sha_weights(bundle.model.model)
    sae_sha = _sha_weights(bundle.sae)
    if model_sha != m0["model_state_sha256"] or sae_sha != m0["sae_state_sha256"]:
        raise RuntimeError("Loaded model or SAE does not match actual M0 weight fingerprints")
    site = ActivationSite(layer=3, hook_name="blocks.3.hook_resid_post")
    if bundle.site.layer != site.layer or bundle.site.hook_name != site.hook_name:
        raise RuntimeError("Loaded checkpoint is incompatible with M0 site")
    tokenizer = bundle.model.model.tokenizer
    token_data = {"vocab": tokenizer.get_vocab(),
                  "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
                  "chat_template": getattr(tokenizer, "chat_template", None)}
    tokenizer_sha = fingerprint(token_data)
    cache_id = ActivationIdentity(model_key="pythia70m", model_state_sha256=model_sha,
                                  tokenizer_sha256=tokenizer_sha,
                                  hook_name=site.hook_name, layer=site.layer,
                                  preprocessing={"source": "M0 SimpleCopying token IDs", "selection": "last_valid"},
                                  execution={"model_backend": bundle.model.backend,
                                             "run_mode": "eval", "device": "cpu"})
    records = [SampleRecord(sample_id=row["sample_id"], token_ids=tuple(row["token_ids"]),
                            selection=TokenSelection("last_valid"), split="m0-reference",
                            prompt_group_id=row["sample_id"])
               for row in m0["samples"]]
    if not records:
        raise RuntimeError("M0 reference contains no sample rows")
    # Include two explicitly selected positions to validate multi-position capture.
    first = records[0]
    records.append(SampleRecord(sample_id="m1-multi-position", token_ids=first.token_ids,
                                selection=TokenSelection("indices", (0, len(first.token_ids) - 1)),
                                split="m1-extra", prompt_group_id=first.sample_id))
    with ActivationWriter(cache_root, cache_id, width=512,
                          shard_target_bytes=4096) as writer:
        stats = collect_activations(bundle.model, records, writer, site=site)
        raw_path = writer.path
    if stats.collected != len(records):
        raise RuntimeError("M1 real run did not collect the requested activations")
    raw = ShardedReader(raw_path)
    max_probe_err = 0.0
    for m0row in m0["samples"]:
        stored = raw.get(m0row["sample_id"])
        assert stored.positions == (len(m0row["token_ids"]) - 1,)
        probe = np.asarray(stored.values[0, :len(m0row["activation_probe"])])
        err = float(np.max(np.abs(probe - np.array(m0row["activation_probe"]))))
        max_probe_err = max(max_probe_err, err)
        if not np.allclose(probe, m0row["activation_probe"], atol=1e-4, rtol=1e-4):
            raise RuntimeError("ActivationStore does not reproduce M0 activation probe")
    full_tokens = torch.tensor([list(first.token_ids)], dtype=torch.long)
    multi = raw.get("m1-multi-position")
    assert len(multi.positions) == 2
    for row, position in zip(multi.values, multi.positions):
        expected = bundle.model.capture(tokens=full_tokens,
                                        site=ActivationSite(site.layer, site.hook_name, position))
        if not np.allclose(row, expected.detach().cpu().numpy(), atol=1e-5, rtol=1e-5):
            raise RuntimeError("M1 multi-position capture differs from M0 adapter")
    # All requested samples already in the store: no model forward is required.
    def no_model_calls(*args, **kwargs):
        raise AssertionError("resuming an existing activation cache must not run the model")
    original_capture = bundle.model.model.run_with_cache
    bundle.model.model.run_with_cache = no_model_calls
    try:
        with ActivationWriter(cache_root, cache_id, width=512) as writer:
            resumed = collect_activations(bundle.model, records, writer, site=site)
            if resumed.collected != 0 or resumed.reused != len(records):
                raise RuntimeError("activation resume did not reuse every committed sample")
        cident = CodeIdentity(sae_key="pythia70m_resid_sm", sae_state_sha256=sae_sha,
                              d_in=512, d_sae=32768, inference_mode="per_token")
        code_path, encoded = encode_from_store(raw_path, bundle.sae, cident, layout="csr",
                                               max_rows=16, shard_target_bytes=256*1024)
        code_path2, repeat = encode_from_store(raw_path, bundle.sae, cident, layout="csr",
                                               max_rows=16, shard_target_bytes=256*1024)
    finally:
        bundle.model.model.run_with_cache = original_capture
    if code_path != code_path2 or encoded.collected != len(records) or repeat.reused != len(records):
        raise RuntimeError("SAE code cache did not resume independently from model activations")
    encoded_cache = ShardedReader(code_path)
    nonzero = []
    for m0row in m0["samples"]:
        actual = encoded_cache.get(m0row["sample_id"]).values[0]
        nz = int(np.count_nonzero(actual))
        nonzero.append(nz)
        if nz != int(m0row["codes_l0"]) or int(np.argmax(np.abs(actual))) != m0row["selected_feature"]:
            raise RuntimeError("SAE code cache disagrees with real M0 feature activations")
    report = {"schema": "saebase.m1.real_cache.v1", "real_checkpoint_validated": True,
              "m0_reference_verified": True, "status": "passed",
              "model_state_sha256": model_sha, "sae_state_sha256": sae_sha,
              "activation_key": cache_id.key, "sae_code_key": cident.key,
              "sample_count": raw.sample_count, "row_count": raw.row_count,
              "code_sample_count": encoded_cache.sample_count,
              "activation_shards": len(raw.manifest["shards"]),
              "code_shards": len(encoded_cache.manifest["shards"]),
              "max_m0_activation_probe_abs_error": max_probe_err,
              "model_forward_calls_during_resume_and_sae_encoding": 0,
              "resumed_samples": resumed.reused, "encoded_samples": encoded.collected,
              "repeat_code_cache_reused": repeat.reused,
              "code_l0_for_m0_samples": nonzero,
              "cache_root": "M1 local artifact (not committed to source)"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("[M1] Real activation cache, M0 reference, multi-position capture and SAE reuse passed")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-reference", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.m0_reference, args.cache_root, args.output)


if __name__ == "__main__":
    main()
