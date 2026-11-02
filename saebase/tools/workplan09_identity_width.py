#!/usr/bin/env python3
"""Separate identity-replay width from candidate regressions.

The pilot baseline forward uses sample_batch=8 and the old identity replay used
microbatch=4. This script reports both widths on the current candidate. Prompt
text is not written.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from saebase.api import ModelAdapter  # noqa: E402
from saebase.interventions import BatchEngine, Condition, ResultWriter  # noqa: E402
from saebase.api.sae_loading import load_model_and_sae  # noqa: E402
from saebase.store import sha256_state_dict  # noqa: E402
from workplan09_gpu_pilot import (  # noqa: E402
    TOLERANCES,
    _build_token_rows,
    _experiment,
    _samples_from_token_rows,
    _tokenizer_sha,
    dtype_name,
)

SITES = (
    {"sae_key": "pythia410m_mlp_topk", "layer": 4, "model_key": "pythia410m"},
    {
        "sae_key": "r1_distill_qwen15b_resid_l20",
        "layer": 20,
        "model_key": "r1_distill_qwen15b",
    },
)


def max_abs(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    if not left:
        return 0.0
    return max(float((a.float() - b.float()).abs().max().item()) for a, b in zip(left, right))


def passes(left: list[torch.Tensor], right: list[torch.Tensor], atol: float) -> bool:
    return all(
        torch.allclose(a.float(), b.float(), rtol=1e-5, atol=atol) for a, b in zip(left, right)
    )


def measure(spec: dict[str, object]) -> dict[str, object]:
    raw, sae, meta = load_model_and_sae(str(spec["sae_key"]), int(spec["layer"]), device="cuda")
    adapter = ModelAdapter(raw)
    model_dtype = dtype_name(next(raw.parameters()).dtype)
    atol = TOLERANCES[model_dtype]["atol"]
    rows, pad = _build_token_rows(raw)
    samples = _samples_from_token_rows(rows)
    group = samples[:8]
    identity = _experiment(
        adapter,
        sae,
        spec,
        sha256_state_dict(raw),
        sha256_state_dict(sae),
        _tokenizer_sha(raw),
    )
    engine = BatchEngine(adapter, sae, identity, pad_token_id=pad)
    snaps, logits = engine._baselines(group)
    pairs = [(sample, snap.activation) for sample, snap in zip(group, snaps)]
    same = engine._patched_logits(pairs)
    half = len(group) // 2
    narrow = engine._patched_logits(pairs[:half])
    lengths = [sample.valid_length for sample in group]
    row: dict[str, object] = {
        "model_key": spec["model_key"],
        "backend": adapter.backend,
        "model_dtype": model_dtype,
        "atol": atol,
        "n": len(group),
        "lengths": lengths,
        "hook": meta.get("hook_name"),
        "same_width_max_abs": max_abs(logits, same),
        "same_width_pass": passes(logits, same, atol),
        "cross_width_max_abs": max_abs(logits[:half], narrow),
        "cross_width_pass": passes(logits[:half], narrow, atol),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    with tempfile.TemporaryDirectory() as tmp:
        with ResultWriter(tmp, identity) as writer:
            stats = engine.run(
                group,
                [Condition("identity", "identity")],
                writer,
                sample_batch=8,
                microbatch=4,
                tolerance=atol,
                verify_identity=True,
            )
        row["engine_identity"] = "PASS"
        row["baseline_batches"] = stats.baseline_batches
        row["patch_batches"] = stats.patch_batches
    del raw, sae, adapter, engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    report = []
    for spec in SITES:
        try:
            row = measure(spec)
        except RuntimeError as exc:
            row = {
                "model_key": spec["model_key"],
                "engine_identity": "FAIL",
                "error": str(exc)[:500],
            }
        report.append(row)
        print(json.dumps(row), flush=True)
    out = Path(
        os.environ.get("WP09_RUN", "/workspace/experiments/performance/workplan09/rtx5070-20261021")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity_width.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
