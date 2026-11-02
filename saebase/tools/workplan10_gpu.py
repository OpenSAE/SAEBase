#!/usr/bin/env python3
"""Work Plan 10 RTX 5070 check. Does not write prompt text or token ids."""

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

OUT = Path("/workspace/experiments/performance/workplan10/rtx5070-20261021")
ARCHIVED_BASE = {
    "qwen": 5.58690007776022,
    "pythia": 2.3632943271659315,
    "gemma": 2.898293502163142,
    "r1": 10.02117465576157,
}


def _nonzero(codes: torch.Tensor) -> list[int]:
    return [int(i) for i in torch.nonzero(codes != 0, as_tuple=False).flatten().tolist()]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


class _MemoryWriter:
    """In-process journal. Model-only timing does not hash or touch the disk."""

    def __init__(self, identity) -> None:
        self.identity = identity
        self._index: set[str] = set()

    def has(self, key: str) -> bool:
        return key in self._index

    def append(self, row) -> bool:
        self._index.add(row["measurement_id"])
        return True

    def flush(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def _time_mode(
    adapter,
    sae,
    identity,
    samples,
    conditions,
    pad,
    tolerance,
    session,
    release: bool,
    mode: str,
    disk: bool,
):
    from saebase.interventions import BatchEngine, ResultWriter

    times = []
    peaks = []
    forwards = []
    for rep in range(6):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        engine = BatchEngine(
            adapter,
            sae,
            identity,
            pad_token_id=pad,
            identity_lease=session,
            release_content_check=release,
        )
        if disk:
            folder = OUT / "tmp-time" / mode / str(rep)
            if folder.exists():
                import shutil

                shutil.rmtree(folder)
            writer_cm = ResultWriter(folder, identity)
        else:
            writer_cm = _MemoryWriter(identity)
        with writer_cm as writer:
            engine.run(
                samples,
                conditions,
                writer,
                sample_batch=8,
                microbatch=4,
                verify_identity=False,
                tolerance=tolerance,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        peaks.append(int(torch.cuda.max_memory_allocated()))
        forwards.append(dict(engine.counters))
    warmed = times[1:]
    median = sorted(warmed)[len(warmed) // 2]
    last = forwards[-1]
    return {
        "mode": mode,
        "warmup_seconds": times[0],
        "repetitions_seconds": warmed,
        "median_seconds": median,
        "peak_allocated_bytes": max(peaks[1:]),
        "model_forwards": int(last["model_forwards"]),
        "combined_capture_forwards": int(last["combined_capture_forwards"]),
        "separate_baseline_forwards": int(last["separate_baseline_forwards"]),
        "sae_encodes": int(last["sae_encodes"]),
        "patch_forwards": int(last["patch_forwards"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=tuple(ARCHIVED_BASE), required=True)
    args = parser.parse_args()
    import workplan09_gpu_pilot as pilot
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.api.features import _decode, snapshot_from_activation
    from saebase.interventions import BatchEngine, ResultWriter, apply_condition
    from saebase.api.sae_loading import load_model_and_sae
    from saebase.api.sae_profiles import get_profile
    from saebase.store import VerifiedCheckpointSession, sha256_state_dict

    specs = {
        "qwen": (pilot.SITES[1], 1e-2),
        "pythia": (pilot.SITES[0], 1e-5),
        "gemma": (pilot.SITES[2], 1e-5),
        "r1": (pilot.SUBSTITUTE, 1e-5),
    }
    spec, tolerance = specs[args.site]
    raw, sae, _meta = load_model_and_sae(spec["sae_key"], int(spec["layer"]), device="cuda")
    adapter = ModelAdapter(raw)
    rows, pad = pilot._build_token_rows(raw)
    samples = pilot._samples_from_token_rows(rows)
    hook = get_profile(spec["sae_key"]).hook_name(int(spec["layer"]))
    first = samples[0]
    site0 = ActivationSite(int(spec["layer"]), hook, first.patch_position)
    snapshot = snapshot_from_activation(
        sae,
        tokens=adapter.tokens(tokens=torch.tensor([first.tokens], dtype=torch.long)),
        site=site0,
        activation=adapter.capture(
            tokens=adapter.tokens(tokens=torch.tensor([first.tokens])), site=site0
        ),
    )
    conditions = pilot._conditions_for(_nonzero(snapshot.codes), int(sae.cfg.d_sae))
    model_sha = sha256_state_dict(raw)
    sae_sha = sha256_state_dict(sae)
    identity = pilot._experiment(adapter, sae, spec, model_sha, sae_sha, "c" * 64)
    parity = []
    if adapter.supports_capture_with_logits() and adapter.backend == "hf_hooks":
        for sample in samples:
            ids = adapter.tokens(tokens=torch.tensor([sample.tokens], dtype=torch.long))
            site = ActivationSite(int(spec["layer"]), hook, sample.patch_position)
            separate_h = adapter.capture(tokens=ids, site=site)
            separate_y = adapter.forward_logits(tokens=ids, readout_position=sample.readout_index)
            combined_h, combined_y = adapter.capture_with_logits(
                tokens=ids, site=site, readout_position=sample.readout_index
            )
            z_separate = sae.encode(separate_h.unsqueeze(0))[0]
            z_combined = sae.encode(combined_h.unsqueeze(0))[0]
            flips = sorted(set(_nonzero(z_separate)).symmetric_difference(_nonzero(z_combined)))
            parity.append(
                {
                    "sample_id": sample.record.sample_id,
                    "activation_max_abs": _max_abs(combined_h, separate_h),
                    "logits_max_abs": _max_abs(combined_y, separate_y),
                    "feature_flips": len(flips),
                }
            )
    edit_rows = []
    for sample in samples[:8]:
        ids = adapter.tokens(tokens=torch.tensor([sample.tokens], dtype=torch.long))
        site = ActivationSite(int(spec["layer"]), hook, sample.patch_position)
        snap = snapshot_from_activation(
            sae, tokens=ids, site=site, activation=adapter.capture(tokens=ids, site=site)
        )
        for condition in conditions:
            patch = apply_condition(sae, snap, condition)
            if condition.operation == "identity":
                expected_h = snap.activation
            elif condition.operation == "reconstruction":
                expected_h = snap.reconstruction
            else:
                before = _decode(
                    sae, snap.codes, int(sae.cfg.d_in), ln_mu=snap.ln_mu, ln_std=snap.ln_std
                )
                after = _decode(
                    sae, patch.codes, int(sae.cfg.d_in), ln_mu=snap.ln_mu, ln_std=snap.ln_std
                )
                expected_h = snap.activation + (after - before).to(dtype=snap.activation.dtype)
            expected_y = adapter.patch_logits(
                tokens=ids,
                site=site,
                activation=expected_h,
                readout_position=sample.readout_index,
            )
            edited_y = adapter.patch_logits(
                tokens=ids,
                site=site,
                activation=patch.activation,
                readout_position=sample.readout_index,
            )
            edit_rows.append(
                {
                    "sample_id": sample.record.sample_id,
                    "condition": condition.name,
                    "activation_max_abs": _max_abs(patch.activation, expected_h),
                    "logits_max_abs": _max_abs(edited_y, expected_y),
                }
            )
    session = VerifiedCheckpointSession.mint(raw, sae)
    identity_status = "PASS"
    identity_error = None
    try:
        folder = OUT / args.site / "identity"
        if folder.exists():
            import shutil

            shutil.rmtree(folder)
        engine = BatchEngine(
            adapter,
            sae,
            identity,
            pad_token_id=pad,
            identity_lease=session,
            release_content_check=False,
        )
        with ResultWriter(folder, identity) as writer:
            engine.run(
                samples,
                conditions,
                writer,
                sample_batch=8,
                microbatch=4,
                verify_identity=True,
                tolerance=tolerance,
            )
        session.strict_recheck()
    except RuntimeError as exc:
        identity_status = "FAIL"
        identity_error = str(exc)[:300]
    torch.cuda.synchronize()
    hash_started = time.perf_counter()
    sha256_state_dict(raw)
    sha256_state_dict(sae)
    torch.cuda.synchronize()
    strict_hash_seconds = time.perf_counter() - hash_started
    warm = _time_mode(
        adapter,
        sae,
        identity,
        samples,
        conditions,
        pad,
        tolerance,
        session,
        False,
        "warm-lease",
        True,
    )
    strict = _time_mode(
        adapter, sae, identity, samples, conditions, pad, tolerance, None, False, "strict", True
    )
    model_only = _time_mode(
        adapter,
        sae,
        identity,
        samples,
        conditions,
        pad,
        tolerance,
        session,
        False,
        "model-only",
        False,
    )
    report = {
        "site": args.site,
        "backend": adapter.backend,
        "torch": torch.__version__,
        "n_samples": len(samples),
        "n_conditions": len(conditions),
        "lengths": sorted({int(sample.valid_length) for sample in samples}),
        "model_sha256": model_sha,
        "sae_sha256": sae_sha,
        "capture_parity": {
            "n": len(parity),
            "worst_activation_max_abs": max(
                (row["activation_max_abs"] for row in parity), default=None
            ),
            "worst_logits_max_abs": max((row["logits_max_abs"] for row in parity), default=None),
            "feature_flips": sum(row["feature_flips"] for row in parity),
        },
        "edit_parity": {
            "n": len(edit_rows),
            "worst_activation_max_abs": max(
                (row["activation_max_abs"] for row in edit_rows), default=None
            ),
            "worst_logits_max_abs": max((row["logits_max_abs"] for row in edit_rows), default=None),
        },
        "identity_gate": identity_status,
        "identity_gate_error": identity_error,
        "strict_hash_seconds": strict_hash_seconds,
        "warm_lease": warm,
        "strict": strict,
        "model_only": model_only,
        "archived_base_median_seconds": ARCHIVED_BASE[args.site],
        "warm_ratio_versus_archived_base": ARCHIVED_BASE[args.site] / warm["median_seconds"],
        "strict_ratio_versus_archived_base": ARCHIVED_BASE[args.site] / strict["median_seconds"],
        "model_only_ratio_versus_archived_base": ARCHIVED_BASE[args.site]
        / model_only["median_seconds"],
    }
    destination = OUT / args.site
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    parity_lines = [json.dumps({"kind": "capture", **row}) for row in parity]
    parity_lines.extend(json.dumps({"kind": "edit", **row}) for row in edit_rows)
    (destination / "parity.jsonl").write_text(
        ("\n".join(parity_lines) + "\n") if parity_lines else ""
    )
    timing_rows = []
    for block in (warm, strict, model_only):
        timing_rows.append(
            json.dumps(
                {
                    "site": args.site,
                    "mode": block["mode"],
                    "warmup_seconds": block["warmup_seconds"],
                    "repetitions_seconds": block["repetitions_seconds"],
                    "median_seconds": block["median_seconds"],
                    "peak_allocated_bytes": block["peak_allocated_bytes"],
                    "model_forwards": block["model_forwards"],
                    "patch_forwards": block["patch_forwards"],
                    "combined_capture_forwards": block["combined_capture_forwards"],
                    "separate_baseline_forwards": block["separate_baseline_forwards"],
                    "sae_encodes": block["sae_encodes"],
                }
            )
        )
    (destination / "before_after.jsonl").write_text("\n".join(timing_rows) + "\n")
    failure = []
    if identity_error:
        failure.append({"site": args.site, "kind": "identity_gate", "error": identity_error})
    (destination / "failures.jsonl").write_text(
        ("\n".join(json.dumps(row) for row in failure) + "\n") if failure else ""
    )
    print(json.dumps(report, indent=2))
    failed = identity_status != "PASS"
    if parity and (
        report["capture_parity"]["feature_flips"]
        or report["capture_parity"]["worst_logits_max_abs"] > tolerance
    ):
        failed = True
    if edit_rows and report["edit_parity"]["worst_logits_max_abs"] > tolerance:
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
