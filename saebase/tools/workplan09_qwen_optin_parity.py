#!/usr/bin/env python3
"""Engine-level parity for opt-in HF run_batch after the row-wise kernel fix.

Compares enable_hf_masked_batch True against the default single-example path
on equal-length and mixed-length groups. Does not change the default.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from saebase.api import ModelAdapter  # noqa: E402
from saebase.interventions import BatchEngine, Condition, ResultWriter  # noqa: E402
from saebase.sae.loading import load_model_and_sae  # noqa: E402
from saebase.store import sha256_state_dict  # noqa: E402
from workplan09_gpu_pilot import (  # noqa: E402
    TOLERANCES,
    _build_token_rows,
    _experiment,
    _samples_from_token_rows,
    _tokenizer_sha,
    dtype_name,
)

SPEC = {
    "sae_key": "qwen3_1_7b_base_resid_w32k",
    "layer": 7,
    "model_key": "qwen3_1_7b_base",
}


def ids_of(codes: torch.Tensor) -> set[int]:
    return set(int(i) for i in torch.nonzero(codes != 0, as_tuple=False).flatten().tolist())


def main() -> None:
    raw, sae, meta = load_model_and_sae(SPEC["sae_key"], SPEC["layer"], device="cuda")
    adapter = ModelAdapter(raw)
    model_dtype = dtype_name(next(raw.parameters()).dtype)
    atol = TOLERANCES[model_dtype]["atol"]
    rtol = TOLERANCES[model_dtype]["rtol"]
    rows, pad = _build_token_rows(raw)
    samples = _samples_from_token_rows(rows)
    finals = [
        s
        for s in samples
        if s.record.sample_id.endswith("-final") and s.record.prompt_group_id != "wp09-eq"
    ]
    equal = [s for s in samples if s.record.prompt_group_id == "wp09-eq"]
    identity = _experiment(
        adapter,
        sae,
        SPEC,
        sha256_state_dict(raw),
        sha256_state_dict(sae),
        _tokenizer_sha(raw),
    )
    serial = BatchEngine(adapter, sae, identity, pad_token_id=pad, enable_hf_masked_batch=False)
    opted = BatchEngine(adapter, sae, identity, pad_token_id=pad, enable_hf_masked_batch=True)
    groups = {
        "equal_length": equal,
        "mixed_length": finals[:8],
    }
    report: dict[str, object] = {
        "backend": adapter.backend,
        "dtype": model_dtype,
        "hook": meta.get("hook_name"),
        "default_flag": serial.enable_hf_masked_batch,
        "groups": [],
    }
    for name, chosen in groups.items():
        if not chosen:
            continue
        serial_snaps, serial_logits = serial._baselines(chosen)
        opt_snaps, opt_logits = opted._baselines(chosen)
        flips = 0
        act_max = 0.0
        logit_max = 0.0
        for left, right, y0, y1 in zip(serial_snaps, opt_snaps, serial_logits, opt_logits):
            if ids_of(left.codes) != ids_of(right.codes):
                flips += 1
            act_max = max(
                act_max, float((left.activation.float() - right.activation.float()).abs().max())
            )
            logit_max = max(logit_max, float((y0.float() - y1.float()).abs().max()))
        act_ok = all(
            torch.allclose(a.activation.float(), b.activation.float(), rtol=rtol, atol=atol)
            for a, b in zip(serial_snaps, opt_snaps)
        )
        logit_ok = all(
            torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
            for a, b in zip(serial_logits, opt_logits)
        )
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            with ResultWriter(tmp, identity) as writer:
                stats = opted.run(
                    chosen,
                    [Condition("identity", "identity")],
                    writer,
                    sample_batch=len(chosen),
                    microbatch=4,
                    tolerance=atol,
                    verify_identity=True,
                )
        elapsed = time.perf_counter() - started
        group = {
            "name": name,
            "n": len(chosen),
            "lengths": [s.valid_length for s in chosen],
            "feature_id_flips": flips,
            "activation_max_abs": act_max,
            "logits_max_abs": logit_max,
            "activation_allclose": act_ok,
            "logits_allclose": logit_ok,
            "identity": "PASS",
            "baseline_batches": stats.baseline_batches,
            "seconds": elapsed,
        }
        report["groups"].append(group)
        print(json.dumps(group), flush=True)
    out = Path(
        os.environ.get("WP09_RUN", "/workspace/experiments/performance/workplan09/rtx5070-20261021")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwen_optin_parity.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
