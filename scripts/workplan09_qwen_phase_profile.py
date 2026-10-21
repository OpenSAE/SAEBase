#!/usr/bin/env python3
"""Phase timing for one serial Qwen engine run. No prompt text or token ids."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

OUT = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021/gpu06")


def _sync_time(fn, bucket: dict[str, float | int], key: str):
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        bucket[key] = float(bucket[key]) + (time.perf_counter() - started)
        bucket[key + "_n"] = int(bucket[key + "_n"]) + 1
        return result

    return wrapped


def main() -> None:
    import workplan09_gpu_pilot as pilot
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions import BatchEngine, ResultWriter
    from saebase.sae.loading import load_model_and_sae
    from saebase.store import sha256_state_dict

    raw, sae, _meta = load_model_and_sae("qwen3_1_7b_base_resid_w32k", 7, device="cuda")
    adapter = ModelAdapter(raw)
    rows, pad = pilot._build_token_rows(raw)
    samples = pilot._samples_from_token_rows(rows)
    site = ActivationSite(7, "blocks.7.hook_resid_post", samples[0].patch_position)
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
    identity = pilot._experiment(
        adapter,
        sae,
        pilot.SITES[1],
        sha256_state_dict(raw),
        sha256_state_dict(sae),
        "a" * 64,
    )
    buckets: dict[str, float | int] = {
        "forward": 0.0,
        "forward_n": 0,
        "encode": 0.0,
        "encode_n": 0,
        "decode": 0.0,
        "decode_n": 0,
        "apply": 0.0,
        "apply_n": 0,
        "metrics": 0.0,
        "metrics_n": 0,
        "append": 0.0,
        "append_n": 0,
    }
    raw._model.forward = _sync_time(raw._model.forward, buckets, "forward")
    sae.encode = _sync_time(sae.encode, buckets, "encode")
    sae.decode = _sync_time(sae.decode, buckets, "decode")
    import saebase.interventions.engine as engine_mod

    engine_mod.apply_condition = _sync_time(engine_mod.apply_condition, buckets, "apply")
    engine_mod._metrics = _sync_time(engine_mod._metrics, buckets, "metrics")
    original_append = ResultWriter.append
    ResultWriter.append = _sync_time(original_append, buckets, "append")
    folder = OUT / "profile-phases"
    if folder.exists():
        import shutil

        shutil.rmtree(folder)
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine = BatchEngine(adapter, sae, identity, pad_token_id=pad)
    with ResultWriter(folder, identity) as writer:
        stats = engine.run(
            samples,
            conditions,
            writer,
            sample_batch=8,
            microbatch=4,
            verify_identity=False,
            tolerance=1e-2,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    named = ("forward", "encode", "decode", "apply", "metrics", "append")
    accounted = sum(float(buckets[name]) for name in named)
    # apply includes decode, so do not add decode again into the exclusive total.
    exclusive = accounted - float(buckets["decode"])
    report = {
        "wall_seconds": wall,
        "measured": int(stats.measured),
        "buckets": buckets,
        "exclusive_accounted_seconds": exclusive,
        "unaccounted_seconds": wall - exclusive,
        "decode_impl": type(sae.decode).__name__
        if not hasattr(sae.decode, "__func__")
        else "method",
        "sae_class": type(sae).__name__,
    }
    (OUT / "qwen_phase_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
