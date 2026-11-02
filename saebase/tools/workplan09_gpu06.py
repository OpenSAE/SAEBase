#!/usr/bin/env python3
"""Work Plan 09 GPU-06: interp-engine/vLLM and reference steering on one RTX 5070.

Reference steering uses the SAEBase full-prefix oracle. Accelerated backends are
experimental and are not enabled by default. Raw prompt text is not written to
the public report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN = Path(
    os.environ.get("WP09_RUN", "/workspace/experiments/performance/workplan09/rtx5070-20261021")
)
PROMPT = "The capital of France is"
MAX_NEW = 4
SITE = {"sae_key": "qwen3_1_7b_base_resid_w32k", "layer": 7, "model_key": "qwen3_1_7b_base"}
FALLBACK = {"sae_key": "pythia410m_mlp_topk", "layer": 4, "model_key": "pythia410m"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def logit_summary(logits: Any) -> dict[str, Any]:
    import torch

    vector = logits.detach().float().cpu().reshape(-1)
    top = torch.topk(vector, k=min(5, vector.numel()))
    return {
        "max_abs": float(vector.abs().max().item()),
        "argmax": int(torch.argmax(vector).item()),
        "top_ids": [int(i) for i in top.indices.tolist()],
        "top_logits": [float(v) for v in top.values.tolist()],
    }


def cmd_reference() -> int:
    """Greedy full-prefix SAEBase steering, plus an unsteered HF generation."""
    import torch
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions import Condition, apply_condition
    from saebase.api.sae_loading import load_model_and_sae

    out = RUN / "gpu06"
    out.mkdir(parents=True, exist_ok=True)
    spec = SITE
    try:
        raw, sae, meta = load_model_and_sae(spec["sae_key"], spec["layer"], device="cuda")
    except Exception as exc:  # noqa: BLE001
        spec = FALLBACK
        try:
            raw, sae, meta = load_model_and_sae(spec["sae_key"], spec["layer"], device="cuda")
        except Exception as second:  # noqa: BLE001
            write_json(
                out / "reference_steering.json",
                {
                    "status": "BLOCKED",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "fallback_error": f"{type(second).__name__}: {second}"[:500],
                },
            )
            return 2
    adapter = ModelAdapter(raw)
    ids = [int(x) for x in raw.to_tokens(PROMPT)[0].tolist()]
    hook = meta["hook_name"]
    layer = int(spec["layer"])
    condition = Condition("ablate", "ablate", (0,))
    steps = []
    current = list(ids)
    for step in range(MAX_NEW):
        tokens = torch.tensor([current], dtype=torch.long, device="cuda")
        position = len(current) - 1
        site = ActivationSite(layer, hook, position)
        clean = adapter.forward_logits(tokens=tokens, readout_position=-1)
        snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
        if step == 0:
            nonzero = [
                int(i)
                for i in torch.nonzero(snapshot.codes != 0, as_tuple=False).flatten().tolist()
            ]
            feature = nonzero[0] if nonzero else 0
            condition = Condition("ablate", "ablate", (feature,))
            torch.save(
                {
                    "last_resid": snapshot.activation.detach().float().cpu(),
                    "token_ids": list(current),
                    "dtype": str(snapshot.activation.dtype).replace("torch.", ""),
                    "feature_id": feature,
                },
                out / "reference_last_resid.pt",
            )
        patch = apply_condition(sae, snapshot, condition)
        edited = adapter.patch_logits(
            tokens=tokens,
            site=site,
            activation=patch.activation,
            readout_position=-1,
        )
        delta = (edited.float() - clean.float()).abs()
        nxt = int(torch.argmax(clean.float().cpu()).item())
        steps.append(
            {
                "step": step,
                "protocol": "saebase_full_prefix_recompute",
                "context_length": len(current),
                "patch_position": position,
                "feature_id": condition.feature_ids[0],
                "clean": logit_summary(clean),
                "edited": logit_summary(edited),
                "max_abs_logit_delta": float(delta.max().item()),
                "next_clean_token": nxt,
            }
        )
        current.append(nxt)
    # Unsteered generation through the same model, greedy, no SAE edit.
    unsteered = []
    plain = list(ids)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(MAX_NEW):
        logits = adapter.forward_logits(
            tokens=torch.tensor([plain], dtype=torch.long, device="cuda"),
            readout_position=-1,
        )
        nxt = int(torch.argmax(logits.float().cpu()).item())
        unsteered.append(nxt)
        plain.append(nxt)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "utc": utc_now(),
        "status": "PASS",
        "site": spec,
        "loaded_model_name": meta.get("loaded_model_name"),
        "hook_name": hook,
        "backend": adapter.backend,
        "protocol": "saebase_full_prefix_recompute",
        "prompt_group": "wp09-gpu06-reference",
        "prompt_token_count": len(ids),
        "new_tokens": MAX_NEW,
        "clean_continuation": [row["next_clean_token"] for row in steps],
        "unsteered_hf_continuation": unsteered,
        "unsteered_seconds": elapsed,
        "unsteered_tokens_per_second": MAX_NEW / elapsed if elapsed else None,
        "cached_steering": {
            "status": "BLOCKED",
            "reason": (
                "No protocol-equivalent KV-cached steering implementation is in this "
                "PR. Persisting edited KV state would be a different protocol and is "
                "not reported as a speedup."
            ),
        },
        "steps": steps,
    }
    write_json(out / "reference_steering.json", report)
    for row in steps:
        append_jsonl(RUN / "steering_steps.jsonl", row)
    print(
        json.dumps(
            {
                "reference": "PASS",
                "site": spec["sae_key"],
                "tokens_per_s": report["unsteered_tokens_per_second"],
            }
        )
    )
    return 0


def cmd_backends() -> int:
    """Install and exercise interp-engine eager/vLLM in a separate environment."""
    env_dir = Path("/workspace/venvs/wp09-interp")
    out = RUN / "gpu06"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "interp_install.log"
    py = env_dir / "bin" / "python"
    if not py.exists():
        subprocess.check_call(["uv", "venv", str(env_dir), "--python", "/venv/main/bin/python"])
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = "/workspace/.cache/uv"
    env["TMPDIR"] = "/workspace/tmp"
    env["HF_HOME"] = "/workspace/.hf_home"
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(py), "interp-engine[vllm]==1.14.0"],
        env=env,
        text=True,
        capture_output=True,
    )
    log.write_text(install.stdout + "\n" + install.stderr, encoding="utf-8")
    if install.returncode != 0:
        # Eager-only is still a real backend attempt if the vLLM extra cannot resolve.
        eager = subprocess.run(
            ["uv", "pip", "install", "--python", str(py), "interp-engine==1.14.0"],
            env=env,
            text=True,
            capture_output=True,
        )
        log.write_text(
            log.read_text() + "\n--- eager retry ---\n" + eager.stdout + eager.stderr,
            encoding="utf-8",
        )
        if eager.returncode != 0:
            write_json(
                out / "backend_attempt.json",
                {
                    "status": "BLOCKED",
                    "reason": "interp-engine install failed",
                    "vllm_extra_returncode": install.returncode,
                    "eager_returncode": eager.returncode,
                    "stderr_tail": (install.stderr + eager.stderr)[-2000:],
                },
            )
            print(json.dumps({"backends": "BLOCKED", "stage": "install"}))
            return 2
    worker = Path(__file__).with_name("workplan09_gpu06_worker.py")
    completed = subprocess.run([str(py), str(worker)], env=env, text=True)
    return completed.returncode


def cmd_align() -> int:
    """Compare the eager residual capture with the SAEBase reference capture."""
    import torch

    ref_path = RUN / "gpu06" / "reference_last_resid.pt"
    eager_path = RUN / "gpu06" / "eager_last_resid.pt"
    if not ref_path.exists() or not eager_path.exists():
        write_json(
            RUN / "gpu06" / "capture_align.json",
            {"status": "BLOCKED", "reason": "reference or eager residual was not saved"},
        )
        print(json.dumps({"align": "BLOCKED"}))
        return 2
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    eager = torch.load(eager_path, map_location="cpu", weights_only=False)
    same_tokens = ref["token_ids"] == eager["token_ids"]
    left = ref["last_resid"].float().reshape(-1)
    right = eager["last_resid"].float().reshape(-1)
    if left.shape != right.shape or not same_tokens:
        report = {
            "status": "BLOCKED",
            "reason": "token ids or residual width differ, so the captures are not the same forward",
            "same_token_ids": same_tokens,
            "reference_width": int(left.numel()),
            "eager_width": int(right.numel()),
            "reference_token_count": len(ref["token_ids"]),
            "eager_token_count": len(eager["token_ids"]),
        }
    else:
        diff = (left - right).abs()
        denom = left.norm().item()
        report = {
            "status": "PASS" if float(diff.max().item()) <= 1e-2 else "FAIL",
            "atol": 1e-2,
            "rtol": 1e-2,
            "max_abs": float(diff.max().item()),
            "rel_l2": float(diff.norm().item() / denom) if denom else None,
            "reference_dtype": ref.get("dtype"),
            "eager_dtype": eager.get("dtype"),
            "feature_id": ref.get("feature_id"),
        }
    report["utc"] = utc_now()
    write_json(RUN / "gpu06" / "capture_align.json", report)
    print(json.dumps({"align": report["status"], "max_abs": report.get("max_abs")}))
    return 0 if report["status"] == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("reference", "backends", "align"))
    args = parser.parse_args()
    if args.cmd == "reference":
        root = os.environ.get("SAEBASE_ROOT", "/workspace/SAEBase-wp09-candidate")
        sys.path.insert(0, root)
        raise SystemExit(cmd_reference())
    if args.cmd == "align":
        raise SystemExit(cmd_align())
    raise SystemExit(cmd_backends())


if __name__ == "__main__":
    main()
