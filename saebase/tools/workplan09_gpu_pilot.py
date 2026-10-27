#!/usr/bin/env python3
"""Work Plan 09 focused RTX 5070 pilot for PR #44.

Measures the pinned SAEBASE_ROOT checkout. Raw prompts and token ids stay under
the private run directory. This is an engineering pilot, not a 48-site sweep
and not a substitute for the Stage D 64-group population.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Frozen before any speed or parity result is inspected.
SITES = (
    {
        "sae_key": "pythia410m_mlp_topk",
        "layer": 4,
        "model_key": "pythia410m",
        "architecture": "topk",
        "reason": "Smallest open verified Pythia TopK MLP site; TransformerLens backend.",
    },
    {
        "sae_key": "qwen3_1_7b_base_resid_w32k",
        "layer": 7,
        "model_key": "qwen3_1_7b_base",
        "architecture": "topk",
        "reason": "Open verified Qwen residual TopK site; HF hooks backend and masked batch path.",
    },
    {
        "sae_key": "gemma270m_scope_resid",
        "layer": 5,
        "model_key": "gemma270m",
        "architecture": "jumprelu",
        "reason": "Verified Gemma Scope JumpReLU residual site; second architecture.",
    },
)
SUBSTITUTE = {
    "sae_key": "r1_distill_qwen15b_resid_l20",
    "layer": 20,
    "model_key": "r1_distill_qwen15b",
    "architecture": "topk",
    "reason": (
        "Registered open verified site used only if gemma270m weights are unavailable. "
        "Does not replace a JumpReLU result."
    ),
    "replaces": "gemma270m_scope_resid:L5",
}
PROMPTS = (
    ("wp09-g0", "The capital of France is"),
    ("wp09-g1", "One plus one equals"),
    ("wp09-g2", "A short note."),
    ("wp09-g3", "After the rain the street was quiet and the lamps were on"),
    ("wp09-g4", "Select the larger number: 3 or 9."),
    ("wp09-g5", "Hello"),
    ("wp09-g6", "The experiment records the next token without changing the prompt text."),
    ("wp09-g7", "Left right left right stop"),
)
TOLERANCES = {
    "float32": {"atol": 1e-5, "rtol": 1e-4},
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
}
BUDGETS = {
    "max_wall_seconds": 10800,
    "max_disk_growth_gb": 80,
    "max_download_gb": 40,
    "max_resident_vram_gb": 11.5,
    "one_model_at_a_time": True,
    "warmup_reps": 1,
    "timed_reps": 5,
    "full_48_site_sweep": False,
    "gpu05": "not_executed",
}
BASE_SHA = "339fbb603bc6d999fd126a91ea8709fee1c599c2"
CANDIDATE_SHA = "db8e652d818ea2ec6995ebe87b9a001d623db2bb"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_root() -> Path:
    return Path(
        os.environ.get("WP09_RUN", "/workspace/experiments/performance/workplan09/rtx5070-20261021")
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def site_id(spec: dict[str, Any]) -> str:
    return f"{spec['model_key']}:{spec['sae_key']}:L{spec['layer']}"


def bootstrap() -> Path:
    root = Path(os.environ["SAEBASE_ROOT"]).resolve()
    sys.path.insert(0, str(root))
    return root


def dtype_name(tensor_dtype: Any) -> str:
    text = str(tensor_dtype).replace("torch.", "")
    if text not in TOLERANCES:
        raise RuntimeError(f"unsupported dtype for the frozen tolerance table: {text}")
    return text


def numeric_errors(actual: Any, expected: Any) -> dict[str, float]:
    import torch

    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-12)
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel": float((delta / denom).max().item()) if delta.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0,
    }


def within(actual: Any, expected: Any, tol: dict[str, float]) -> bool:
    import torch

    return bool(
        torch.allclose(actual.float(), expected.float(), atol=tol["atol"], rtol=tol["rtol"])
    )


def nonzero_ids(codes: Any) -> list[int]:
    import torch

    flat = codes.detach().float().reshape(-1)
    return [int(i) for i in torch.nonzero(flat != 0, as_tuple=False).flatten().tolist()]


def gpu_snapshot() -> dict[str, Any]:
    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_bytes": int(props.total_memory),
        "free_vram_bytes": int(free_b),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved()),
        "allocated_vram_bytes": int(torch.cuda.memory_allocated()),
    }


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024**3), 2)


def cmd_preflight() -> None:
    import torch

    root = run_root()
    private = root / "private_inputs"
    private.mkdir(parents=True, exist_ok=True)
    (private / "prompts.json").write_text(
        json.dumps(
            {
                "population": "wp09-pilot",
                "not_stage_d_64": True,
                "not_research_lab_500": True,
                "note": (
                    "Stage D raw prompt text is not in this checkout. These eight "
                    "original strings are a separate pilot identity and must not be pooled."
                ),
                "prompts": [{"group_id": gid, "text": text} for gid, text in PROMPTS],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        "utc": utc_now(),
        "hostname_class": "vast-container",
        "os": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_snapshot() if torch.cuda.is_available() else None,
        "disk_free_gb_workspace": disk_free_gb(Path("/workspace")),
        "disk_free_gb_root": disk_free_gb(Path("/")),
        "ram_available_kb": _mem_available_kb(),
        "base_sha": BASE_SHA,
        "candidate_sha": CANDIDATE_SHA,
        "ci_run": "https://github.com/OpenSAE/SAEBase/actions/runs/39546669785",
        "ci_at_candidate_sha": {
            "offline-tests (3.11)": "SUCCESS",
            "offline-tests (3.12)": "SUCCESS",
            "analysis-install": "SUCCESS",
            "package": "SUCCESS",
            "localization-native": "SUCCESS",
            "quality": "SUCCESS",
            "research-lab-corpus-cpu": "SKIPPED",
        },
        "venv": sys.prefix,
        "hf_token_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
    }
    for name in ("transformers", "sae-lens", "transformer-lens", "eai-sparsify"):
        try:
            import importlib.metadata as metadata

            env[name] = metadata.version(name)
        except Exception as exc:  # noqa: BLE001
            env[name] = f"unavailable: {type(exc).__name__}"
    write_json(root / "environment.json", env)
    write_json(
        root / "site_plan.json",
        {
            "frozen_before_measurement": True,
            "utc": utc_now(),
            "sites": list(SITES),
            "substitute_if_blocked": SUBSTITUTE,
            "prompts_are_not_stage_d": True,
            "group_ids": [gid for gid, _ in PROMPTS],
            "equal_length_rule": "first 6 token ids when a prompt has at least 6 tokens; group wp09-eq",
            "patch_positions": "last valid token on every group; index 1 when length >= 4",
            "tolerances": TOLERANCES,
            "speed_gate_median_warmed_throughput": 1.15,
        },
    )
    write_json(
        root / "run_manifest.json",
        {
            "utc_start": utc_now(),
            "budgets": BUDGETS,
            "tolerances_recorded_before_runs": TOLERANCES,
            "base_worktree": "/workspace/SAEBase-wp09-base",
            "candidate_worktree": "/workspace/SAEBase-wp09-candidate",
            "base_sha": BASE_SHA,
            "candidate_sha": CANDIDATE_SHA,
            "pr": "https://github.com/OpenSAE/SAEBase/pull/44",
            "population": "wp09-pilot",
            "seeds": {"torch": 7, "oom": 0},
            "quantization": "not_used",
        },
    )
    print(json.dumps({"preflight": "written", "run": str(root)}))


def _mem_available_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _tokenizer_sha(model: Any) -> str:
    from saebase.store import fingerprint

    tokenizer = model.tokenizer
    payload = {
        "vocab": tokenizer.get_vocab(),
        "special": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    return fingerprint(payload)


def _pad_id(tokenizer: Any, used: set[int]) -> int:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0 and value not in used:
            return value
    if 0 not in used:
        return 0
    return max(used) + 1


def _samples_from_token_rows(rows: list[dict[str, Any]]) -> list[Any]:
    from saebase.interventions import MeasurementSample
    from saebase.store import SampleRecord, TokenSelection

    samples = []
    for row in rows:
        mode = row["selection_mode"]
        selection = (
            TokenSelection("indices", tuple(row["indices"]))
            if mode == "indices"
            else TokenSelection("last_valid")
        )
        samples.append(
            MeasurementSample(
                SampleRecord(
                    row["sample_id"],
                    tuple(row["token_ids"]),
                    selection=selection,
                    prompt_group_id=row["group_id"],
                    split="wp09-pilot",
                ),
                readout_position=int(row["readout_position"]),
                target_token_id=int(row["target_token_id"]),
            )
        )
    return samples


def _build_token_rows(model: Any) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    tokenized: list[tuple[str, tuple[int, ...]]] = []
    for group_id, text in PROMPTS:
        ids = tuple(int(x) for x in model.to_tokens(text)[0].tolist())
        tokenized.append((group_id, ids))
        used.update(ids)
    for group_id, ids in tokenized:
        rows.append(
            {
                "sample_id": f"{group_id}-final",
                "group_id": group_id,
                "token_ids": list(ids),
                "selection_mode": "last_valid",
                "indices": [],
                "readout_position": -1,
                "target_token_id": ids[-1],
                "length": len(ids),
            }
        )
        if len(ids) >= 4:
            rows.append(
                {
                    "sample_id": f"{group_id}-early-readout-final",
                    "group_id": group_id,
                    "token_ids": list(ids),
                    "selection_mode": "indices",
                    "indices": [1],
                    "readout_position": -1,
                    "target_token_id": ids[-1],
                    "length": len(ids),
                }
            )
            rows.append(
                {
                    "sample_id": f"{group_id}-early-readout-early",
                    "group_id": group_id,
                    "token_ids": list(ids),
                    "selection_mode": "indices",
                    "indices": [1],
                    "readout_position": 1,
                    "target_token_id": ids[1],
                    "length": len(ids),
                }
            )
    equal = [ids[:6] for _, ids in tokenized if len(ids) >= 6][:4]
    for index, ids in enumerate(equal):
        rows.append(
            {
                "sample_id": f"wp09-eq-{index}",
                "group_id": "wp09-eq",
                "token_ids": list(ids),
                "selection_mode": "last_valid",
                "indices": [],
                "readout_position": -1,
                "target_token_id": ids[-1],
                "length": len(ids),
            }
        )
    pad = _pad_id(model.tokenizer, used)
    return rows, pad


def _conditions_for(feature_ids: list[int], d_sae: int) -> list[Any]:
    from saebase.interventions import Condition

    first = feature_ids[0] if feature_ids else 0
    second = feature_ids[1] if len(feature_ids) > 1 else (1 if first == 0 and d_sae > 1 else 0)
    if second == first:
        second = 0 if first != 0 else min(1, d_sae - 1)
    pair = (first, second) if first != second else (first,)
    retained = (first,)
    return [
        Condition("identity", "identity"),
        Condition("ablate", "ablate", (first,)),
        Condition("scale", "scale", (first,), 0.5),
        Condition("clamp", "clamp", (first,), 0.0),
        Condition("add", "add", (first,), 1.0),
        Condition("coalition", "coalition", pair, retained_ids=retained),
        Condition("reconstruction", "reconstruction"),
    ]


def _load_pair(spec: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from saebase.sae.loading import load_model_and_sae

    free_b, _ = torch.cuda.mem_get_info()
    if free_b < 2 * 1024**3:
        raise RuntimeError(f"refusing to load with only {free_b} bytes free VRAM")
    model, sae, meta = load_model_and_sae(spec["sae_key"], int(spec["layer"]), device="cuda")
    return model, sae, meta


def _release(*objects: Any) -> None:
    import gc
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _capture_store(
    model_adapter: Any,
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    model_sha: str,
    tokenizer_sha: str,
    width: int,
    store_root: Path,
) -> Path:
    from saebase.api import ActivationSite
    from saebase.sae.registry import get_profile
    from saebase.store import ActivationIdentity, ActivationWriter, collect_activations

    hook = get_profile(spec["sae_key"]).hook_name(int(spec["layer"]))
    identity = ActivationIdentity(
        model_key=spec["model_key"],
        model_state_sha256=model_sha,
        tokenizer_sha256=tokenizer_sha,
        hook_name=hook,
        layer=int(spec["layer"]),
        dtype="float32",
        preprocessing={"population": "wp09-pilot", "selection": "explicit"},
        execution={"device": "cuda", "run_mode": "eval"},
    )
    samples = _samples_from_token_rows(rows)
    records = []
    for sample in samples:
        # Multi-position rows are separate single-position samples. That matches
        # M2 and still records both the final token and an earlier position.
        records.append(sample.record)
    site = ActivationSite(int(spec["layer"]), hook)
    with ActivationWriter(store_root, identity, width=width) as writer:
        stats = collect_activations(model_adapter, records, writer, site=site)
        path = writer.path
    if stats.collected + stats.reused != len(records):
        raise RuntimeError(
            f"activation capture collected {stats.collected} and reused {stats.reused} of {len(records)}"
        )
    return path


def _encode_store(
    activation_path: Path, sae: Any, spec: dict[str, Any], sae_sha: str, d_in: int, d_sae: int
) -> dict[str, Any]:
    from saebase.store import CodeIdentity, ShardedReader, encode_from_store

    identity = CodeIdentity(
        sae_key=spec["sae_key"],
        sae_state_sha256=sae_sha,
        d_in=d_in,
        d_sae=d_sae,
        inference_mode="per_token",
    )
    path, stats = encode_from_store(activation_path, sae, identity, max_rows=8)
    again_path, again = encode_from_store(activation_path, sae, identity, max_rows=8)
    reader = ShardedReader(path)
    return {
        "path": str(path),
        "again_path": str(again_path),
        "written": stats.collected,
        "rows": stats.rows_collected,
        "reused_on_second_call": again.reused,
        "sample_count": reader.sample_count,
        "identity_key": identity.key,
    }


def _serial_and_batched_codes(sae: Any, matrix: Any, batch_sizes: list[int]) -> dict[str, Any]:
    import torch

    serial = []
    with torch.inference_mode():
        for row in matrix:
            serial.append(sae.encode(row.unsqueeze(0)))
    serial_codes = torch.cat(serial, dim=0)
    reports = []
    sae_dtype = dtype_name(matrix.dtype if False else next(sae.parameters()).dtype)
    tol = TOLERANCES[sae_dtype]
    for size in batch_sizes:
        if size > matrix.shape[0]:
            reps = (size + matrix.shape[0] - 1) // matrix.shape[0]
            block = matrix.repeat((reps, 1))[:size]
            expected = serial_codes.repeat((reps, 1))[:size]
            kind = "repeated_real_rows"
        else:
            block = matrix[:size]
            expected = serial_codes[:size]
            kind = "distinct_real_rows"
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                batched = sae.encode(block)
            peak = int(torch.cuda.max_memory_allocated())
            err = numeric_errors(batched, expected)
            ids_serial = nonzero_ids(expected)
            ids_batch = nonzero_ids(batched)
            reports.append(
                {
                    "batch_rows": size,
                    "kind": kind,
                    "status": "PASS"
                    if within(batched, expected, tol) and ids_serial == ids_batch
                    else "FAIL",
                    "errors": err,
                    "feature_id_set_equal": ids_serial == ids_batch,
                    "serial_nonzero": len(ids_serial),
                    "batched_nonzero": len(ids_batch),
                    "peak_allocated_bytes": peak,
                    "tolerance": tol,
                    "sae_dtype": sae_dtype,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            reports.append(
                {"batch_rows": size, "kind": kind, "status": "BLOCKED", "reason": str(exc)[:300]}
            )
            break
    return {"rows": int(matrix.shape[0]), "reports": reports}


def _time_encode(sae: Any, matrix: Any, batch: int, reps: int) -> dict[str, Any]:
    import torch

    times = []
    peaks = []
    for _ in range(BUDGETS["warmup_reps"] + reps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            sae.encode(matrix)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        peaks.append(int(torch.cuda.max_memory_allocated()))
    warmed = times[BUDGETS["warmup_reps"] :]
    warmed_sorted = sorted(warmed)
    median = warmed_sorted[len(warmed_sorted) // 2]
    return {
        "batch_rows": batch,
        "warmup_seconds": times[0],
        "repetitions_seconds": warmed,
        "median_seconds": median,
        "min_seconds": min(warmed),
        "max_seconds": max(warmed),
        "rows_per_second": len(matrix) / median if median else None,
        "peak_allocated_bytes": max(peaks[BUDGETS["warmup_reps"] :]),
    }


def _experiment(
    model_adapter: Any,
    sae: Any,
    spec: dict[str, Any],
    model_sha: str,
    sae_sha: str,
    tokenizer_sha: str,
) -> Any:
    from saebase.api import ActivationSite
    from saebase.interventions import ExperimentIdentity
    from saebase.sae.registry import get_profile

    hook = get_profile(spec["sae_key"]).hook_name(int(spec["layer"]))
    return ExperimentIdentity(
        model_sha,
        sae_sha,
        ActivationSite(int(spec["layer"]), hook),
        model_key=spec["model_key"],
        sae_key=spec["sae_key"],
        tokenizer_sha256=tokenizer_sha,
        backend=model_adapter.backend,
    )


def _run_engine(
    model_adapter: Any,
    sae: Any,
    identity: Any,
    samples: list[Any],
    conditions: list[Any],
    out_dir: Path,
    pad_token_id: int,
    tolerance: float,
    verify_identity: bool = True,
) -> dict[str, Any]:
    from dataclasses import asdict

    from saebase.interventions import BatchEngine, ResultWriter

    recorded: dict[str, Any] = {"baselines": [], "patches": []}
    engine = BatchEngine(model_adapter, sae, identity, pad_token_id=pad_token_id)
    original_base = engine._baselines
    original_patch = engine._patched_logits

    def baselines(group: Any) -> Any:
        snaps, logits = original_base(group)
        recorded["baselines"].append(
            {
                "sample_ids": [sample.record.sample_id for sample in group],
                "logits": [row.detach().float().cpu() for row in logits],
                "codes": [snap.codes.detach().float().cpu() for snap in snaps],
                "activations": [snap.activation.detach().float().cpu() for snap in snaps],
            }
        )
        return snaps, logits

    def patched(jobs: Any) -> Any:
        logits = original_patch(jobs)
        recorded["patches"].append(
            {
                "sample_ids": [job[0].record.sample_id for job in jobs],
                "logits": [row.detach().float().cpu() for row in logits],
            }
        )
        return logits

    engine._baselines = baselines
    engine._patched_logits = patched
    with ResultWriter(out_dir, identity) as writer:
        stats = engine.run(
            samples,
            conditions,
            writer,
            sample_batch=8,
            microbatch=4,
            verify_identity=verify_identity,
            tolerance=tolerance,
        )
    payload = asdict(stats)
    payload["recorded"] = recorded
    return payload


def _hook_count(model: Any, layer: int) -> int | None:
    if hasattr(model, "_layer_module"):
        module = model._layer_module(layer)
        return len(getattr(module, "_forward_hooks", {}))
    return None


def _hf_mask_checks(model: Any, samples: list[Any], layer: int) -> list[dict[str, Any]]:
    import torch

    rows = []
    if not callable(getattr(model, "run_batch", None)):
        return [{"check": "run_batch", "status": "BLOCKED", "reason": "model has no run_batch"}]
    chosen = samples[:3]
    width = max(sample.valid_length for sample in chosen)
    ids = torch.zeros((len(chosen), width), dtype=torch.long, device=model.cfg.device)
    mask = torch.zeros_like(ids)
    positions = []
    readouts = []
    for index, sample in enumerate(chosen):
        ids[index, : sample.valid_length] = torch.tensor(sample.tokens, device=ids.device)
        mask[index, : sample.valid_length] = 1
        positions.append(sample.patch_position)
        readouts.append(sample.readout_index)
    before = _hook_count(model, layer)
    activations, logits = model.run_batch(
        ids,
        layer=layer,
        positions=positions,
        readout_positions=readouts,
        attention_mask=mask,
    )
    after = _hook_count(model, layer)
    from saebase.api import ActivationSite, ModelAdapter

    adapter = ModelAdapter(model)
    hook = f"blocks.{layer}.hook_resid_post"
    mismatches = []
    for index, sample in enumerate(chosen):
        single_ids = torch.tensor([sample.tokens], dtype=torch.long, device=ids.device)
        expected_h = adapter.capture(
            tokens=single_ids, site=ActivationSite(layer, hook, sample.patch_position)
        )
        expected_logits = adapter.forward_logits(
            tokens=single_ids, readout_position=sample.readout_index
        )
        tol = TOLERANCES[dtype_name(expected_logits.dtype)]
        ok_h = within(activations[index], expected_h, tol)
        ok_y = within(logits[index], expected_logits, tol)
        mismatches.append(
            {
                "sample_id": sample.record.sample_id,
                "activation": numeric_errors(activations[index], expected_h),
                "logits": numeric_errors(logits[index], expected_logits),
                "status": "PASS" if ok_h and ok_y else "FAIL",
                "tolerance": tol,
            }
        )
    rows.append(
        {
            "check": "mixed_or_equal_batch_matches_single",
            "status": "PASS" if all(item["status"] == "PASS" for item in mismatches) else "FAIL",
            "hooks_before": before,
            "hooks_after": after,
            "samples": mismatches,
        }
    )
    try:
        model.run_batch(
            ids,
            layer=layer,
            positions=positions,
            readout_positions=readouts,
            attention_mask=1 - mask,
        )
        rows.append({"check": "left_padding_rejected", "status": "FAIL", "reason": "no exception"})
    except ValueError as exc:
        rows.append({"check": "left_padding_rejected", "status": "PASS", "error": str(exc)[:200]})
    try:
        model.run_batch(
            ids,
            layer=layer,
            positions=[-1] * len(chosen),
            readout_positions=readouts,
            attention_mask=mask,
        )
        rows.append({"check": "invalid_position_rejected", "status": "FAIL"})
    except ValueError as exc:
        rows.append(
            {"check": "invalid_position_rejected", "status": "PASS", "error": str(exc)[:200]}
        )
    bad = activations.clone()
    bad[0, 0] = float("nan")
    try:
        model.run_batch(
            ids,
            layer=layer,
            positions=positions,
            readout_positions=readouts,
            attention_mask=mask,
            patches=bad,
        )
        rows.append({"check": "nonfinite_patch_rejected", "status": "FAIL"})
    except ValueError as exc:
        rows.append(
            {"check": "nonfinite_patch_rejected", "status": "PASS", "error": str(exc)[:200]}
        )
    try:
        model._layer_module(10**6)
        rows.append({"check": "missing_layer_rejected", "status": "FAIL"})
    except (ValueError, IndexError) as exc:
        rows.append({"check": "missing_layer_rejected", "status": "PASS", "error": str(exc)[:200]})
    original = model._model.forward

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected model exception")

    model._model.forward = explode
    before_exc = _hook_count(model, layer)
    try:
        model.run_batch(
            ids,
            layer=layer,
            positions=positions,
            readout_positions=readouts,
            attention_mask=mask,
        )
        rows.append({"check": "exception_removes_hook", "status": "FAIL", "reason": "no exception"})
    except RuntimeError:
        after_exc = _hook_count(model, layer)
        rows.append(
            {
                "check": "exception_removes_hook",
                "status": "PASS" if after_exc == before_exc else "FAIL",
                "hooks_before": before_exc,
                "hooks_after": after_exc,
            }
        )
    finally:
        model._model.forward = original
    return rows


def _role_dir(spec: dict[str, Any], role: str) -> Path:
    return run_root() / "sites" / site_id(spec).replace(":", "_") / role


def _shared_dir(spec: dict[str, Any]) -> Path:
    return run_root() / "sites" / site_id(spec).replace(":", "_") / "shared"


def cmd_role(spec: dict[str, Any], role: str) -> int:
    import torch
    from saebase.api import ModelAdapter
    from saebase.store import sha256_state_dict

    destination = _role_dir(spec, role)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    try:
        load_started = time.perf_counter()
        raw_model, sae, meta = _load_pair(spec)
        load_seconds = time.perf_counter() - load_started
        adapter = ModelAdapter(raw_model)
        model_sha = sha256_state_dict(raw_model)
        sae_sha = sha256_state_dict(sae)
        tokenizer_sha = _tokenizer_sha(raw_model)
        model_dtype = dtype_name(next(raw_model.parameters()).dtype)
        sae_dtype = dtype_name(next(sae.parameters()).dtype)
        shared = _shared_dir(spec)
        shared.mkdir(parents=True, exist_ok=True)
        if role == "candidate":
            rows, pad = _build_token_rows(raw_model)
            write_json(shared / "samples.json", {"rows": rows, "pad_token_id": pad})
        else:
            payload = json.loads((shared / "samples.json").read_text(encoding="utf-8"))
            rows, pad = payload["rows"], int(payload["pad_token_id"])
        samples = _samples_from_token_rows(rows)
        from saebase.api import ActivationSite
        from saebase.sae.registry import get_profile

        hook = get_profile(spec["sae_key"]).hook_name(int(spec["layer"]))
        # Capture once on the candidate. The base encodes that same store.
        if role == "candidate":
            probe = adapter.capture(
                tokens=torch.tensor([samples[0].tokens], device="cuda"),
                site=ActivationSite(int(spec["layer"]), hook, samples[0].patch_position),
            )
            activation_path = _capture_store(
                adapter,
                rows,
                spec,
                model_sha,
                tokenizer_sha,
                int(probe.shape[-1]),
                shared / "activations",
            )
            write_json(
                shared / "activation_path.json",
                {"path": str(activation_path), "width": int(probe.shape[-1])},
            )
        else:
            activation_path = Path(
                json.loads((shared / "activation_path.json").read_text())["path"]
            )
        from saebase.store import ShardedReader

        reader = ShardedReader(activation_path)
        matrices = [
            torch.tensor(reader.get(sample.record.sample_id).values, dtype=torch.float32)
            for sample in samples
        ]
        matrix = torch.cat(matrices, dim=0).to(device="cuda", dtype=next(sae.parameters()).dtype)
        encode_report = None
        if role == "candidate":
            encode_report = _serial_and_batched_codes(sae, matrix, [1, 8, 32, 128, 256])
            encode_report["timing_batch"] = _time_encode(
                sae, matrix, int(matrix.shape[0]), BUDGETS["timed_reps"]
            )
            serial_only = []
            with torch.inference_mode():
                for row in matrix:
                    serial_only.append(sae.encode(row.unsqueeze(0))[0].detach().float().cpu())
            features = []
            for sample, codes in zip(samples, serial_only[: len(samples)]):
                nz = nonzero_ids(codes)
                features.append(
                    {
                        "sample_id": sample.record.sample_id,
                        "nonzero_feature_ids_head": nz[:8],
                        "l0": len(nz),
                        "max_abs": float(codes.abs().max().item()) if codes.numel() else 0.0,
                    }
                )
            write_json(
                shared / "frozen_features.json",
                {
                    "source": "candidate per-row sae.encode",
                    "frozen_before_cross_version_timing": True,
                    "samples": features,
                },
            )
        copied = destination / "activation_copy"
        if copied.exists():
            shutil.rmtree(copied)
        shutil.copytree(activation_path, copied, ignore=shutil.ignore_patterns("codes"))
        store_encode = _encode_store(
            copied,
            sae,
            spec,
            sae_sha,
            int(sae.cfg.d_in),
            int(sae.cfg.d_sae),
        )
        feature_map = {
            item["sample_id"]: item["nonzero_feature_ids_head"]
            for item in json.loads((shared / "frozen_features.json").read_text())["samples"]
        }
        # One condition list per sample would change BatchEngine's cross product.
        # Use the first sample's observed support for the shared condition list,
        # and record every sample's own support for the feature-flip check.
        first_ids = feature_map[samples[0].record.sample_id]
        conditions = _conditions_for(first_ids, int(sae.cfg.d_sae))
        identity = _experiment(adapter, sae, spec, model_sha, sae_sha, tokenizer_sha)
        tol = TOLERANCES[model_dtype]
        collection_tolerance = tol["atol"]
        identity_error = None
        try:
            stats = _run_engine(
                adapter,
                sae,
                identity,
                samples,
                conditions,
                destination / "results",
                pad,
                tol["atol"],
                verify_identity=True,
            )
            identity_status = "PASS"
        except RuntimeError as exc:
            if "identity patch" not in str(exc):
                raise
            # Keep the frozen protocol as FAIL, but still collect the remaining
            # edits. 1.0 is only the engine's internal abort threshold.
            identity_status = "FAIL"
            identity_error = str(exc)[:500]
            collection_tolerance = 1.0
            stats = _run_engine(
                adapter,
                sae,
                identity,
                samples,
                conditions,
                destination / "results_collected",
                pad,
                collection_tolerance,
                verify_identity=False,
            )
            stats["identity_gate_error"] = identity_error
        timing = _time_engine(
            adapter, sae, identity, samples, conditions, pad, collection_tolerance, destination
        )
        mask_checks = (
            _hf_mask_checks(raw_model, samples, int(spec["layer"]))
            if adapter.backend == "hf_hooks"
            else [{"check": "hf_mask", "status": "not_applicable", "backend": adapter.backend}]
        )
        summary = {
            "status": "measured",
            "utc": started,
            "role": role,
            "site_id": site_id(spec),
            "code_sha": _head_sha(),
            "load_seconds": load_seconds,
            "backend": adapter.backend,
            "model_dtype": model_dtype,
            "sae_dtype": sae_dtype,
            "model_sha256": model_sha,
            "sae_sha256": sae_sha,
            "tokenizer_sha256": tokenizer_sha,
            "loaded_model_name": meta.get("loaded_model_name"),
            "sae_meta": {
                key: meta.get(key)
                for key in (
                    "repo_id",
                    "sae_release",
                    "sae_id",
                    "hook_name",
                    "loader",
                    "d_in",
                    "d_sae",
                    "folder",
                )
            },
            "gpu_after_load": gpu_snapshot(),
            "n_samples": len(samples),
            "n_conditions": len(conditions),
            "pad_token_id": pad,
            "encode_direct": encode_report,
            "encode_store": store_encode,
            "engine": {key: value for key, value in stats.items() if key != "recorded"},
            "identity_gate": identity_status,
            "identity_gate_error": identity_error,
            "protocol_tolerance": tol,
            "collection_tolerance": collection_tolerance,
            "timing": timing,
            "mask_checks": mask_checks,
            "tolerance": tol,
        }
        recorded = stats.pop("recorded")
        _save_recorded(destination / "private_tensors", recorded)
        write_json(destination / "summary.json", summary)
        print(
            json.dumps(
                {
                    "role": role,
                    "site": site_id(spec),
                    "status": "measured",
                    "backend": adapter.backend,
                }
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        status = _load_status(exc)
        write_json(
            destination / "summary.json",
            {
                "status": status,
                "utc": started,
                "role": role,
                "site_id": site_id(spec),
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-4000:],
                "gpu": gpu_snapshot() if torch.cuda.is_available() else None,
            },
        )
        append_jsonl(
            run_root() / "failures.jsonl",
            {
                "utc": utc_now(),
                "site_id": site_id(spec),
                "role": role,
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
        )
        print(
            json.dumps(
                {"role": role, "site": site_id(spec), "status": status, "error": str(exc)[:300]}
            )
        )
        return 2 if status == "BLOCKED" else 1
    finally:
        _release()


def _head_sha() -> str:
    root = os.environ.get("SAEBASE_ROOT", "")
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_status(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(
        token in text
        for token in ("gated", "401", "403", "access to model", "authorized", "cannot access")
    ):
        return "BLOCKED"
    if "out of memory" in text or "cuda out of memory" in text:
        return "BLOCKED"
    return "FAIL"


def _save_recorded(directory: Path, recorded: dict[str, Any]) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    index = []
    blobs: dict[str, Any] = {}
    for kind in ("baselines", "patches"):
        for entry_index, entry in enumerate(recorded[kind]):
            for sample_id, tensor in zip(entry["sample_ids"], entry["logits"]):
                key = f"{kind}_{entry_index}_{sample_id}"
                blobs[key] = tensor
                index.append({"kind": kind, "sample_id": sample_id, "key": key})
            if kind == "baselines":
                for sample_id, tensor in zip(entry["sample_ids"], entry["codes"]):
                    key = f"codes_{entry_index}_{sample_id}"
                    blobs[key] = tensor
                for sample_id, tensor in zip(entry["sample_ids"], entry["activations"]):
                    key = f"act_{entry_index}_{sample_id}"
                    blobs[key] = tensor
    torch.save(blobs, directory / "tensors.pt")
    write_json(directory / "index.json", index)


def _time_engine(
    model_adapter: Any,
    sae: Any,
    identity: Any,
    samples: list[Any],
    conditions: list[Any],
    pad: int,
    tolerance: float,
    destination: Path,
) -> dict[str, Any]:
    import torch
    from saebase.interventions import BatchEngine, ResultWriter

    times = []
    peaks = []
    stats_last = None
    total = BUDGETS["warmup_reps"] + BUDGETS["timed_reps"]
    for rep in range(total):
        folder = destination / f"time-{rep}"
        if folder.exists():
            shutil.rmtree(folder)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        engine = BatchEngine(model_adapter, sae, identity, pad_token_id=pad)
        with ResultWriter(folder, identity) as writer:
            stats_last = engine.run(
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
    warmed = times[BUDGETS["warmup_reps"] :]
    median = sorted(warmed)[len(warmed) // 2]
    measured = int(stats_last.measured) if stats_last is not None else 0
    return {
        "warmup_seconds": times[0],
        "repetitions_seconds": warmed,
        "median_seconds": median,
        "min_seconds": min(warmed),
        "max_seconds": max(warmed),
        "measurements": measured,
        "measurements_per_second": measured / median if median else None,
        "peak_allocated_bytes": max(peaks[BUDGETS["warmup_reps"] :]),
        "baseline_batches": None if stats_last is None else stats_last.baseline_batches,
        "patch_batches": None if stats_last is None else stats_last.patch_batches,
    }


def _load_status_file(spec: dict[str, Any], role: str) -> dict[str, Any]:
    path = _role_dir(spec, role) / "summary.json"
    if not path.exists():
        return {"status": "MISSING"}
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_compare(spec: dict[str, Any]) -> None:
    import torch

    base = _load_status_file(spec, "base")
    candidate = _load_status_file(spec, "candidate")
    row: dict[str, Any] = {
        "utc": utc_now(),
        "site_id": site_id(spec),
        "base_status": base.get("status"),
        "candidate_status": candidate.get("status"),
    }
    if base.get("status") != "measured" or candidate.get("status") != "measured":
        row["status"] = (
            "BLOCKED" if "BLOCKED" in {base.get("status"), candidate.get("status")} else "FAIL"
        )
        append_jsonl(run_root() / "parity.jsonl", row)
        append_jsonl(run_root() / "summary_rows.jsonl", row)
        write_json(_shared_dir(spec) / "comparison.json", row)
        print(json.dumps(row))
        return
    model_dtype = candidate["model_dtype"]
    tol = TOLERANCES[model_dtype]
    base_tensors = torch.load(
        _role_dir(spec, "base") / "private_tensors" / "tensors.pt",
        map_location="cpu",
        weights_only=False,
    )
    cand_tensors = torch.load(
        _role_dir(spec, "candidate") / "private_tensors" / "tensors.pt",
        map_location="cpu",
        weights_only=False,
    )
    shared_keys = sorted(set(base_tensors) & set(cand_tensors))
    worst = {"max_abs": 0.0, "max_rel": 0.0, "l2": 0.0}
    feature_flips = []
    failed = 0
    for key in shared_keys:
        err = numeric_errors(cand_tensors[key], base_tensors[key])
        passed = within(cand_tensors[key], base_tensors[key], tol)
        if key.startswith("codes_"):
            base_ids = set(nonzero_ids(base_tensors[key]))
            cand_ids = set(nonzero_ids(cand_tensors[key]))
            if base_ids != cand_ids:
                feature_flips.append(
                    {
                        "key": key,
                        "only_base": sorted(base_ids - cand_ids)[:12],
                        "only_candidate": sorted(cand_ids - base_ids)[:12],
                    }
                )
                passed = False
        if not passed:
            failed += 1
        for name in worst:
            worst[name] = max(worst[name], err[name])
    speed_base = base["timing"]["measurements_per_second"]
    speed_cand = candidate["timing"]["measurements_per_second"]
    ratio = (speed_cand / speed_base) if speed_base else None
    row.update(
        {
            "status": "PASS" if failed == 0 else "FAIL",
            "compared_tensors": len(shared_keys),
            "failed_tensors": failed,
            "worst_errors": worst,
            "feature_flips": feature_flips[:20],
            "feature_flip_count": len(feature_flips),
            "tolerance": tol,
            "model_dtype": model_dtype,
            "backend": candidate.get("backend"),
            "base_measurements_per_second": speed_base,
            "candidate_measurements_per_second": speed_cand,
            "throughput_ratio_candidate_over_base": ratio,
            "speed_gate": None if ratio is None else ("PASS" if ratio >= 1.15 else "NOT_MET"),
            "base_median_seconds": base["timing"]["median_seconds"],
            "candidate_median_seconds": candidate["timing"]["median_seconds"],
            "base_peak_allocated_bytes": base["timing"]["peak_allocated_bytes"],
            "candidate_peak_allocated_bytes": candidate["timing"]["peak_allocated_bytes"],
            "base_sha": base.get("code_sha"),
            "candidate_sha": candidate.get("code_sha"),
            "model_sha256": candidate.get("model_sha256"),
            "sae_sha256": candidate.get("sae_sha256"),
            "identity_gate_candidate": candidate.get("identity_gate"),
            "identity_gate_base": base.get("identity_gate"),
            "mask_checks": candidate.get("mask_checks"),
            "encode_direct_status": [
                item.get("status")
                for item in (candidate.get("encode_direct") or {}).get("reports", [])
            ],
        }
    )
    append_jsonl(run_root() / "parity.jsonl", row)
    append_jsonl(
        run_root() / "benchmark_runs.jsonl",
        {
            "utc": utc_now(),
            "site_id": site_id(spec),
            "workload": "hf_or_tl_interventions",
            "base": base["timing"],
            "candidate": candidate["timing"],
            "encode_candidate": (candidate.get("encode_direct") or {}).get("timing_batch"),
            "ratio": ratio,
        },
    )
    write_json(_shared_dir(spec) / "comparison.json", row)
    print(
        json.dumps(
            {
                k: row[k]
                for k in (
                    "site_id",
                    "status",
                    "throughput_ratio_candidate_over_base",
                    "failed_tensors",
                    "feature_flip_count",
                )
            }
        )
    )


def _tiny_hf():
    """Small CUDA HF wrapper used only for the injected-OOM engine test."""
    from types import SimpleNamespace

    import torch
    from models.hf_hooks_backend import HFHooksModel

    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 1.25 + 0.25

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=2)
            self.embed = torch.nn.Embedding(32, 4)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            hidden = self.embed(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden))

    class ToyTokenizer:
        bos_token_id = None
        pad_token_id = 0
        eos_token_id = 1
        special_tokens_map: dict[str, Any] = {}
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [((ord(ch) + 1) % 31) + 1 for ch in text] or [1]

        def get_vocab(self) -> dict[str, int]:
            return {str(i): i for i in range(32)}

    wrapper = HFHooksModel(
        Tiny().cuda().eval(), ToyTokenizer(), model_name="tiny-oom", device="cuda"
    )
    return wrapper


def cmd_oom() -> None:
    """Inject one CUDA OOM and confirm the candidate keeps the reduced microbatch."""
    import torch
    from saebase.api import ActivationSite, ModelAdapter
    from saebase.interventions import (
        BatchEngine,
        Condition,
        ExperimentIdentity,
        MeasurementSample,
        ResultWriter,
    )
    from saebase.store import SampleRecord, TokenSelection, sha256_state_dict

    class TinySAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = type("Cfg", (), {"d_in": 4, "d_sae": 4})()
            self.weight = torch.nn.Parameter(torch.eye(4))

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs @ self.weight

        def decode(self, codes: torch.Tensor) -> torch.Tensor:
            return codes @ self.weight

    wrapper = _tiny_hf()
    sae = TinySAE().cuda().eval()
    adapter = ModelAdapter(wrapper)
    identity = ExperimentIdentity(
        sha256_state_dict(wrapper),
        sha256_state_dict(sae),
        ActivationSite(0, "blocks.0.hook_resid_post"),
        model_key="tiny",
        sae_key="tiny",
        backend=adapter.backend,
    )
    samples = []
    for index, ids in enumerate(((1, 2, 3, 4), (5, 6, 7), (8, 9, 2, 3), (4, 5))):
        samples.append(
            MeasurementSample(
                SampleRecord(
                    f"oom-{index}",
                    ids,
                    selection=TokenSelection("last_valid"),
                    prompt_group_id=f"oom-{index}",
                    split="wp09-oom",
                ),
                target_token_id=1,
            )
        )
    conditions = [Condition("identity", "identity"), Condition("ablate", "ablate", (0,))]
    engine = BatchEngine(adapter, sae, identity, pad_token_id=0)
    widths: list[int] = []
    original = engine._patched_logits
    state = {"seen_large": False}

    def patched(jobs: list[Any]) -> Any:
        widths.append(len(jobs))
        if len(jobs) > 2 and not state["seen_large"]:
            state["seen_large"] = True
            raise torch.cuda.OutOfMemoryError("injected deterministic OOM")
        return original(jobs)

    engine._patched_logits = patched
    out = run_root() / "oom" / "injected"
    if out.exists():
        shutil.rmtree(out)
    with ResultWriter(out, identity) as writer:
        stats = engine.run(
            samples, conditions, writer, sample_batch=4, microbatch=8, tolerance=1e-4
        )
    later = widths[widths.index(next(w for w in widths if w > 2)) + 1 :]
    retained = all(width <= 2 for width in later) if later else False
    report = {
        "utc": utc_now(),
        "code_sha": _head_sha(),
        "attempted_widths": widths,
        "retries": stats.microbatch_retries,
        "measured": stats.measured,
        "expected": len(samples) * len(conditions),
        "reduced_width_retained": retained,
        "status": "PASS"
        if stats.measured == len(samples) * len(conditions)
        and stats.microbatch_retries >= 1
        and retained
        else "FAIL",
    }
    write_json(run_root() / "oom" / "injected_report.json", report)
    print(json.dumps(report))
    _release(wrapper, sae, adapter, engine)


def cmd_corrupt() -> None:
    from saebase.store import CacheIntegrityError, ShardedReader

    # Use the first captured activation store if the pilot has created one.
    stores = list((run_root() / "sites").glob("*/shared/activations/*"))
    stores = [path for path in stores if (path / "manifest.json").exists()]
    if not stores:
        write_json(
            run_root() / "oom" / "corrupt_report.json",
            {"status": "BLOCKED", "reason": "no activation store yet"},
        )
        print(json.dumps({"corrupt": "BLOCKED"}))
        return
    source = stores[0]
    copied = run_root() / "oom" / "corrupted_store"
    if copied.exists():
        shutil.rmtree(copied)
    shutil.copytree(source, copied)
    shard = next(copied.rglob("shard-*.npy"))
    data = bytearray(shard.read_bytes())
    data[0] ^= 0xFF
    shard.write_bytes(data)
    try:
        ShardedReader(copied)
        status = "FAIL"
        error = "reader accepted corrupted shard"
    except CacheIntegrityError as exc:
        status = "PASS"
        error = str(exc)[:300]
    report = {"status": status, "store": str(source), "error": error, "utc": utc_now()}
    write_json(run_root() / "oom" / "corrupt_report.json", report)
    print(json.dumps(report))


def _spawn(root: str, args: list[str]) -> int:
    env = os.environ.copy()
    env["SAEBASE_ROOT"] = root
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", "/workspace/.hf_home")
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], env=env)
    return completed.returncode


def cmd_all(base: str, candidate: str) -> int:
    cmd_preflight()
    micro = subprocess.run(
        [
            sys.executable,
            str(Path(candidate) / "scripts" / "benchmark_encoding.py"),
            "--device",
            "cuda",
            "--rows",
            "2048",
            "--width",
            "1024",
            "--features",
            "32768",
            "--repeat",
            "5",
            "--batch-sizes",
            "1,8,32,128,256",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    (run_root() / "encoding_microbench.json").write_text(
        micro.stdout
        if micro.returncode == 0
        else json.dumps(
            {
                "status": "FAIL",
                "returncode": micro.returncode,
                "stderr": micro.stderr[-2000:],
            }
        ),
        encoding="utf-8",
    )
    exit_code = 0
    plan = list(SITES)
    for spec in plan:
        cand_code = _spawn(
            candidate,
            [
                "role",
                "--sae-key",
                spec["sae_key"],
                "--layer",
                str(spec["layer"]),
                "--model-key",
                spec["model_key"],
                "--architecture",
                spec["architecture"],
                "--role",
                "candidate",
            ],
        )
        if cand_code == 2 and spec["sae_key"] == "gemma270m_scope_resid":
            append_jsonl(
                run_root() / "failures.jsonl",
                {
                    "utc": utc_now(),
                    "site_id": site_id(spec),
                    "status": "BLOCKED",
                    "substitution": site_id(SUBSTITUTE),
                },
            )
            spec = SUBSTITUTE
            cand_code = _spawn(
                candidate,
                [
                    "role",
                    "--sae-key",
                    spec["sae_key"],
                    "--layer",
                    str(spec["layer"]),
                    "--model-key",
                    spec["model_key"],
                    "--architecture",
                    spec["architecture"],
                    "--role",
                    "candidate",
                ],
            )
        samples_ready = (_shared_dir(spec) / "samples.json").exists()
        if cand_code != 0 and not samples_ready:
            exit_code = max(exit_code, cand_code)
            continue
        if cand_code != 0:
            exit_code = max(exit_code, cand_code)
        base_code = _spawn(
            base,
            [
                "role",
                "--sae-key",
                spec["sae_key"],
                "--layer",
                str(spec["layer"]),
                "--model-key",
                spec["model_key"],
                "--architecture",
                spec["architecture"],
                "--role",
                "base",
            ],
        )
        exit_code = max(exit_code, base_code)
        if base_code == 0:
            _spawn(
                candidate,
                [
                    "compare",
                    "--sae-key",
                    spec["sae_key"],
                    "--layer",
                    str(spec["layer"]),
                    "--model-key",
                    spec["model_key"],
                    "--architecture",
                    spec["architecture"],
                ],
            )
    # OOM retention is a candidate-code behavior.
    exit_code = max(exit_code, _spawn(candidate, ["oom"]))
    _spawn(candidate, ["corrupt"])
    return exit_code


def _spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sae_key": args.sae_key,
        "layer": args.layer,
        "model_key": args.model_key,
        "architecture": args.architecture,
        "reason": "cli",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    role = sub.add_parser("role")
    for name in (role, sub.add_parser("compare")):
        name.add_argument("--sae-key", required=True)
        name.add_argument("--layer", required=True, type=int)
        name.add_argument("--model-key", required=True)
        name.add_argument("--architecture", default="unspecified")
    role.add_argument("--role", choices=("base", "candidate"), required=True)
    sub.add_parser("oom")
    sub.add_parser("corrupt")
    entire = sub.add_parser("all")
    entire.add_argument("--base", default="/workspace/SAEBase-wp09-base")
    entire.add_argument("--candidate", default="/workspace/SAEBase-wp09-candidate")
    args = parser.parse_args()
    if args.cmd == "preflight":
        cmd_preflight()
        return
    if args.cmd == "all":
        raise SystemExit(cmd_all(args.base, args.candidate))
    if args.cmd in {"role", "compare", "oom"}:
        if args.cmd != "oom":
            os.environ.setdefault("SAEBASE_ROOT", "/workspace/SAEBase-wp09-candidate")
            bootstrap()
        else:
            os.environ.setdefault("SAEBASE_ROOT", "/workspace/SAEBase-wp09-candidate")
            bootstrap()
    if args.cmd == "role":
        raise SystemExit(cmd_role(_spec_from_args(args), args.role))
    if args.cmd == "compare":
        cmd_compare(_spec_from_args(args))
        return
    if args.cmd == "oom":
        cmd_oom()
        return
    if args.cmd == "corrupt":
        os.environ.setdefault("SAEBASE_ROOT", "/workspace/SAEBase-wp09-candidate")
        bootstrap()
        cmd_corrupt()
        return


if __name__ == "__main__":
    main()
