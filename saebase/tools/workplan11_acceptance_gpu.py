"""Mandatory RTX 5070 end-to-end, clean-KV and vLLM SAE validation.

The CPU-only CI tests never invoke this script. All GPU claims must come from
its real per-run JSON and the referenced exact code / checkpoint revisions.

Examples:
 python saebase/tools/workplan11_acceptance_gpu.py e2e
 python saebase/tools/workplan11_acceptance_gpu.py kv --steps 8
 python saebase/tools/workplan11_acceptance_gpu.py kv --steps 32
 python saebase/tools/workplan11_acceptance_gpu.py hf-reference
 # In a separate interp-engine[vllm] environment/process:
 python saebase/tools/workplan11_acceptance_gpu.py vllm --backend vllm
 python saebase/tools/workplan11_acceptance_gpu.py vllm --backend vllm-static
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(os.environ.get("WP11_RUN", "/workspace/experiments/performance/workplan11/final_5070"))
PROMPT = "The capital of France is"
SAE_KEY = "qwen3_1_7b_base_resid_w32k"
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
LAYER = 7


def persist(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"file": str(OUT / name), **data}, default=str))


def synchronize() -> None:
    torch.cuda.synchronize()


def load_pair():
    from saebase.sae.loading import load_model_and_sae

    return load_model_and_sae(SAE_KEY, LAYER, device="cuda")


def e2e() -> None:
    import workplan09_gpu_pilot as pilot
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions import BatchEngine, ResultWriter, read_results
    from saebase.store import VerifiedCheckpointSession, sha256_state_dict
    from saebase.sae.registry import get_profile

    raw, sae, _ = load_pair()
    adapter = ModelAdapter(raw)
    rows, pad = pilot._build_token_rows(raw)
    samples = pilot._samples_from_token_rows(rows)
    first = samples[0]
    site = ActivationSite(LAYER, get_profile(SAE_KEY).hook_name(LAYER), first.patch_position)
    snap = capture_features(
        adapter,
        sae,
        site=site,
        tokens=adapter.tokens(tokens=torch.tensor([first.tokens], dtype=torch.long)),
    )
    active = torch.nonzero(snap.codes != 0).flatten().tolist()
    conditions = pilot._conditions_for(active, int(sae.cfg.d_sae))
    model_sha, sae_sha = sha256_state_dict(raw), sha256_state_dict(sae)
    identity = pilot._experiment(adapter, sae, pilot.SITES[1], model_sha, sae_sha, "b" * 64)
    session = VerifiedCheckpointSession.mint(raw, sae)
    if not callable(getattr(raw, "supports_suffix_replay", None)):
        raise RuntimeError("candidate lacks suffix replay")
    if not raw.supports_suffix_replay():
        raise RuntimeError("registered Qwen backend cannot replay its suffix")
    report = {
        "workload": "25 real selected site rows x 7 edits",
        "samples": len(samples),
        "conditions": len(conditions),
        "model_sha256": model_sha,
        "sae_sha256": sae_sha,
        "runs": {"original": [], "suffix": []},
    }
    by_mode = {}
    with tempfile.TemporaryDirectory(prefix="wp11e2e-") as root:
        for mode in ("original", "suffix"):
            for rep in range(6):
                folder = Path(root) / mode / str(rep)
                synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                engine = BatchEngine(
                    adapter,
                    sae,
                    identity,
                    pad_token_id=pad,
                    identity_lease=session,
                    release_content_check=False,
                    enable_suffix_replay=(mode == "suffix"),
                )
                with ResultWriter(folder, identity) as writer:
                    stats = engine.run(
                        samples,
                        conditions,
                        writer,
                        sample_batch=8,
                        microbatch=4,
                        verify_identity=True,
                        tolerance=1e-2,
                    )
                synchronize()
                elapsed = time.perf_counter() - t0
                if rep == 1:
                    by_mode[mode] = {
                        item["measurement_id"]: item for item in read_results(folder, identity)
                    }
                if rep:
                    report["runs"][mode].append(
                        {
                            "elapsed_seconds": elapsed,
                            "allocated_bytes": torch.cuda.max_memory_allocated(),
                            "reserved_bytes": torch.cuda.max_memory_reserved(),
                            "measured": stats.measured,
                            "counters": engine.counters,
                        }
                    )
    before = by_mode["original"]
    after = by_mode["suffix"]
    if before.keys() != after.keys():
        raise RuntimeError("suffix replay changed the measurement identity set")
    err = 0.0
    changed = 0
    for key in before:
        if before[key]["condition_name"] != after[key]["condition_name"]:
            changed += 1
        metrics_a = before[key]["metrics"]
        metrics_b = after[key]["metrics"]
        for metric, value in metrics_a.items():
            if metric not in metrics_b:
                changed += 1
                continue
            if isinstance(value, (int, float)):
                diff = abs(float(value) - float(metrics_b[metric]))
                err = max(err, diff)
                if diff > 1e-2 + 1e-2 * abs(float(value)):
                    changed += 1
    medians = {}
    for mode in ("original", "suffix"):
        values = sorted(x["elapsed_seconds"] for x in report["runs"][mode])
        medians[mode] = values[len(values) // 2]
    report["median_seconds"] = medians
    report["speedup"] = medians["original"] / medians["suffix"]
    report["max_metric_abs"] = err
    report["changed_metrics"] = changed
    report["parity_pass"] = changed == 0
    report["speed_gate_pass"] = report["speedup"] >= 1.15
    session.strict_recheck()
    persist("end_to_end.json", report)
    if changed:
        raise SystemExit("FAIL: end-to-end scientific parity changed")


def kv(steps: int) -> None:
    from saebase.api import ModelAdapter
    from saebase.interventions.kv_cache_exact import (
        CleanPrefixKVDecoder,
        compare_cached_step_with_reference,
        KVParityError,
    )
    from saebase.interventions.specs import Condition

    raw, sae, _ = load_pair()
    adapter = ModelAdapter(raw)
    ids = raw.to_tokens(PROMPT, prepend_bos=False)
    # Select a genuinely active feature on THIS exact tokenization, not a
    # hard-coded coordinate from a different run or with different BOS rules.
    from saebase.api import ActivationSite, capture_features

    initial_site = ActivationSite(LAYER, f"blocks.{LAYER}.hook_resid_post", -1)
    initial = capture_features(adapter, sae, site=initial_site, tokens=ids)
    selected = torch.nonzero(initial.codes != 0).flatten().tolist()
    if not selected:
        raise RuntimeError("no nonzero SAE feature for clean-KV steering test")
    feature_id = int(selected[0])
    decoder = CleanPrefixKVDecoder(
        raw, sae, layer=LAYER, hook_name=f"blocks.{LAYER}.hook_resid_post"
    )
    decoder.start(ids)
    entries = []
    start = time.perf_counter()
    try:
        for index in range(steps):
            row = compare_cached_step_with_reference(
                decoder,
                adapter,
                sae,
                Condition("ablate", "ablate", (feature_id,)),
                atol=1e-2,
                rtol=1e-2,
            )
            row["step"] = index
            entries.append(row)
            # Fixed deterministic continuation independent of candidate output.
            decoder.append(int(ids[0, -1]))
    except (KVParityError, RuntimeError, ValueError) as exc:
        persist(
            f"clean_kv_parity_{steps}.json",
            {
                "status": "FAIL",
                "steps_completed": len(entries),
                "error": f"{type(exc).__name__}: {exc}",
                "rows": entries,
            },
        )
        raise
    finally:
        decoder.close()
    synchronize()
    parity_time = time.perf_counter() - start

    from saebase.api import ActivationSite, capture_features
    from saebase.interventions.operators import apply_condition
    from saebase.interventions.specs import Condition

    condition = Condition("ablate", "ablate", (feature_id,))
    token = int(ids[0, -1])
    full_times = []
    kv_times = []
    for mode in ("full-prefix", "clean-kv"):
        for rep in range(6):
            current = ids.detach().clone()
            instance = None
            if mode == "clean-kv":
                instance = CleanPrefixKVDecoder(
                    raw, sae, layer=LAYER, hook_name=f"blocks.{LAYER}.hook_resid_post"
                )
                instance.start(current)
            synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                if mode == "full-prefix":
                    site = ActivationSite(
                        LAYER, f"blocks.{LAYER}.hook_resid_post", current.shape[1] - 1
                    )
                    snap = capture_features(adapter, sae, site=site, tokens=current)
                    patch = apply_condition(sae, snap, condition)
                    adapter.patch_logits(tokens=current, site=site, activation=patch.activation)
                else:
                    instance.step(condition)
                if mode == "clean-kv":
                    instance.append(token)
                else:
                    current = torch.cat(
                        (current, torch.tensor([[token]], device=current.device)), dim=1
                    )
            synchronize()
            elapsed = time.perf_counter() - t0
            if rep:
                (full_times if mode == "full-prefix" else kv_times).append(elapsed)
            if instance is not None:
                instance.close()
    full_median = sorted(full_times)[len(full_times) // 2]
    kv_median = sorted(kv_times)[len(kv_times) // 2]
    persist(
        f"clean_kv_parity_{steps}.json",
        {
            "status": "PARITY_PASS",
            "steps": entries,
            "elapsed_with_reference_seconds": parity_time,
            "reference_seconds": full_times,
            "cached_seconds": kv_times,
            "reference_median_seconds": full_median,
            "cached_median_seconds": kv_median,
            "full_prefix_over_cached": full_median / kv_median,
            "speed_gate_pass": full_median / kv_median >= 1.15,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "note": "KV fork deep-copies the clean past before edited calls; promote only after measured gain.",
        },
    )


def hf_reference() -> None:
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions.operators import apply_condition
    from saebase.interventions.specs import Condition
    from saebase.store import sha256_state_dict

    raw, sae, _ = load_pair()
    adapter = ModelAdapter(raw)
    ids = raw.to_tokens(PROMPT, prepend_bos=False)
    site = ActivationSite(LAYER, f"blocks.{LAYER}.hook_resid_post", -1)
    snapshot = capture_features(adapter, sae, site=site, tokens=ids)
    nonzero = torch.nonzero(snapshot.codes != 0).flatten().tolist()
    feature = int(nonzero[0]) if nonzero else 0
    patch = apply_condition(sae, snapshot, Condition("ablate", "ablate", (feature,)))
    clean = adapter.forward_logits(tokens=ids)
    edited = adapter.patch_logits(tokens=ids, site=site, activation=patch.activation)
    OUT.mkdir(parents=True, exist_ok=True)
    # Private file: the token IDs and activation vector must NOT go to GitHub.
    torch.save(
        {
            "ids": ids.cpu(),
            "resid": snapshot.activation.float().cpu(),
            "patched": patch.activation.float().cpu(),
            "feature_id": feature,
            "feature_ids": torch.nonzero(snapshot.codes != 0).flatten().cpu().tolist(),
            "clean_logits": clean.float().cpu(),
            "edited_logits": edited.float().cpu(),
            "model_sha": sha256_state_dict(raw),
            "sae_sha": sha256_state_dict(sae),
        },
        OUT / "hf_reference_private.pt",
    )
    persist(
        "hf_reference_status.json",
        {
            "status": "PASS",
            "model": MODEL_NAME,
            "layer": LAYER,
            "n_tokens": int(ids.shape[1]),
            "note": "raw reference tensors saved privately, not published",
        },
    )


def vllm(backend: str) -> None:
    for extra in os.environ.get("SAEBASE_EXTRA_SITE", "").split(os.pathsep):
        if extra and extra not in sys.path:
            # Later entries fill missing pure-Python deps. Torch and vLLM
            # stay first because this process already imported them.
            sys.path.append(extra)
    from saebase.sae.loading import load_sae_for_profile
    from saebase.interventions.specs import Condition
    from saebase.vllm_backend.sae_interventions import InterpVLLMSAE

    reference = torch.load(OUT / "hf_reference_private.pt", map_location="cpu", weights_only=True)
    sae, _ = load_sae_for_profile(SAE_KEY, LAYER, device="cuda")
    try:
        # Static graphs need a few megabytes past the weight pool. 0.50 left
        # a zero-byte static budget on this 12 GiB card.
        utilization = 0.70 if backend == "vllm-static" else 0.50
        engine = InterpVLLMSAE.load(
            MODEL_NAME,
            sae,
            layer=LAYER,
            backend=backend,
            gpu_memory_utilization=utilization,
            max_model_len=512,
        )
    except Exception as exc:
        persist(
            f"vllm_{backend}.json",
            {"backend": backend, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    try:
        try:
            outcome = engine.measure(
                reference["ids"].flatten().tolist(),
                Condition("ablate", "ablate", (int(reference["feature_id"]),)),
                patch_position=-1,
                atol=1e-2,
                rtol=1e-2,
            )
        except RuntimeError as exc:
            persist(
                f"vllm_{backend}.json",
                {
                    "backend": backend,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "reference_equivalence": "FAIL_SCIENTIFIC_REFERENCE_PARITY",
                },
            )
            raise
        clean_err = float((outcome.clean_residual.float() - reference["resid"]).abs().max().item())
        edited_err = float(
            (outcome.edited_residual.float() - reference["patched"]).abs().max().item()
        )
        clean_logit_err = float(
            (outcome.clean_logits.float() - reference["clean_logits"].float()).abs().max().item()
        )
        edited_logit_err = float(
            (outcome.edited_logits.float() - reference["edited_logits"].float()).abs().max().item()
        )
        feature_support_match = tuple(reference["feature_ids"]) == outcome.feature_ids_before
        numeric_pass = (
            feature_support_match
            and torch.allclose(
                outcome.clean_residual.float(),
                reference["resid"].float(),
                atol=1e-2,
                rtol=1e-2,
            )
            and torch.allclose(
                outcome.edited_residual.float(),
                reference["patched"].float(),
                atol=1e-2,
                rtol=1e-2,
            )
            and torch.allclose(
                outcome.clean_logits.float(),
                reference["clean_logits"].float(),
                atol=1e-2,
                rtol=1e-2,
            )
            and torch.allclose(
                outcome.edited_logits.float(),
                reference["edited_logits"].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )
        data = {
            "backend": backend,
            "status": "REFERENCE_PARITY_PASS" if numeric_pass else "BACKEND_NATIVE_ONLY",
            "input_length": outcome.input_length,
            "position": outcome.patch_position,
            "vllm_internal_edit_max_abs": outcome.activation_error,
            "hf_capture_max_abs": clean_err,
            "hf_edited_activation_max_abs": edited_err,
            "hf_clean_logit_max_abs": clean_logit_err,
            "hf_edited_logit_max_abs": edited_logit_err,
            "feature_support_match": feature_support_match,
            "feature_support": len(outcome.feature_ids_before),
            "reference_equivalence": (
                "FULL_CAPTURE_SAE_EDIT_AND_LOGIT_PASS"
                if numeric_pass
                else "FAIL_SCIENTIFIC_REFERENCE_PARITY"
            ),
            "clean_next_text": outcome.clean_next_text,
            "edited_next_text": outcome.edited_next_text,
        }
        persist(f"vllm_{backend}.json", data)
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("e2e", "kv", "hf-reference", "vllm"))
    parser.add_argument("--steps", type=int, choices=(8, 32), default=8)
    parser.add_argument("--backend", choices=("vllm", "vllm-static"), default="vllm")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("real GPU required")
    if args.mode == "e2e":
        e2e()
    elif args.mode == "kv":
        kv(args.steps)
    elif args.mode == "hf-reference":
        hf_reference()
    else:
        vllm(args.backend)


if __name__ == "__main__":
    main()
