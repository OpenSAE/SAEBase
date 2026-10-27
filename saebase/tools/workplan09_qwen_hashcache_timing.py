#!/usr/bin/env python3
"""Time the serial Qwen workload after the state-hash cache. No prompt text."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

OUT = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021/qwen_hashcache")


def _hash_seconds(fn, module: object) -> tuple[str, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    digest = fn(module)
    torch.cuda.synchronize()
    return digest, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=("qwen", "pythia", "gemma", "r1"), default="qwen")
    parser.add_argument("--skip-identity", action="store_true")
    args = parser.parse_args()
    import workplan09_gpu_pilot as pilot
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions import BatchEngine, ResultWriter
    from saebase.sae.loading import load_model_and_sae
    from saebase.store import sha256_state_dict

    specs = {
        "qwen": (pilot.SITES[1], 1e-2, 5.58690007776022),
        "pythia": (pilot.SITES[0], 1e-5, 2.3632943271659315),
        "gemma": (pilot.SITES[2], 1e-5, 2.898293502163142),
        "r1": (pilot.SUBSTITUTE, 1e-5, 10.02117465576157),
    }
    spec, tolerance, base_median = specs[args.site]
    raw, sae, _meta = load_model_and_sae(spec["sae_key"], int(spec["layer"]), device="cuda")
    adapter = ModelAdapter(raw)
    model_sha, first_model_s = _hash_seconds(sha256_state_dict, raw)
    _, second_model_s = _hash_seconds(sha256_state_dict, raw)
    sae_sha, first_sae_s = _hash_seconds(sha256_state_dict, sae)
    _, second_sae_s = _hash_seconds(sha256_state_dict, sae)
    if second_model_s <= 0 or model_sha != sha256_state_dict(raw):
        raise RuntimeError("model state hash cache did not repeat")
    rows, pad = pilot._build_token_rows(raw)
    samples = pilot._samples_from_token_rows(rows)
    from saebase.sae.registry import get_profile

    hook = get_profile(spec["sae_key"]).hook_name(int(spec["layer"]))
    site = ActivationSite(int(spec["layer"]), hook, samples[0].patch_position)
    snapshot = capture_features(
        adapter,
        sae,
        site=site,
        tokens=adapter.tokens(tokens=torch.tensor([samples[0].tokens], dtype=torch.long)),
    )
    nonzero = [
        int(i) for i in torch.nonzero(snapshot.codes != 0, as_tuple=False).flatten().tolist()
    ]
    conditions = pilot._conditions_for(nonzero, int(sae.cfg.d_sae))
    identity = pilot._experiment(adapter, sae, spec, model_sha, sae_sha, "b" * 64)
    identity_dir = OUT / args.site / "identity"
    if identity_dir.exists():
        import shutil

        shutil.rmtree(identity_dir)
    identity_status = "SKIPPED" if args.skip_identity else "PASS"
    identity_error = None
    if not args.skip_identity:
        try:
            engine = BatchEngine(adapter, sae, identity, pad_token_id=pad)
            with ResultWriter(identity_dir, identity) as writer:
                engine.run(
                    samples,
                    conditions,
                    writer,
                    sample_batch=8,
                    microbatch=4,
                    verify_identity=True,
                    tolerance=tolerance,
                )
        except RuntimeError as exc:
            identity_status = "FAIL"
            identity_error = str(exc)[:300]
    timing = pilot._time_engine(
        adapter,
        sae,
        identity,
        samples,
        conditions,
        pad,
        tolerance,
        OUT / args.site / "timing",
    )
    median = float(timing["median_seconds"])
    report = {
        "model_hash_first_seconds": first_model_s,
        "model_hash_second_seconds": second_model_s,
        "sae_hash_first_seconds": first_sae_s,
        "sae_hash_second_seconds": second_sae_s,
        "identity_gate": identity_status,
        "identity_gate_error": identity_error,
        "median_seconds": median,
        "repetitions_seconds": timing["repetitions_seconds"],
        "peak_allocated_bytes": timing["peak_allocated_bytes"],
        "measurements_per_second": timing["measurements_per_second"],
        "site": args.site,
        "ratio_versus_archived_base": base_median / median if median else None,
        "archived_base_median_seconds": base_median,
    }
    destination = OUT / args.site
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if identity_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
