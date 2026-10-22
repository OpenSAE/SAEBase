"""P4 native SAEBench custom SAE worker; uses original upstream run_evals.

Supports actually loaded registered SAE objects, not hand-made metric
emulation. A bounded subprocess is mandatory before any native suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from saebase.sae.loading import load_sae_for_profile
from saebase.sae.registry import get_profile
from saebase.store import sha256_state_dict


ALLOWED = {"core", "sparse_probing"}
SHA = re.compile(r"^[0-9a-f]{40}$")


def native_worker(
    *, upstream: Path, sae_key: str, layer: int, model_name: str,
    eval_type: str, device: str,
) -> dict[str, Any]:
    """Call original SAEBench's exact custom SAE evaluator in this process."""
    if eval_type not in ALLOWED or device not in {"cpu", "cuda"}:
        raise ValueError("Choose supported original protocol and device")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Real CUDA GPU is unavailable")
    sys.path.insert(0, str(upstream.resolve()))
    from sae_bench.custom_saes.run_all_evals_custom_saes import (  # type: ignore[import-not-found]
        MODEL_CONFIGS, run_evals,
    )

    if model_name not in MODEL_CONFIGS:
        raise ValueError("Original native custom SAE runner does not support this model")
    profile = get_profile(sae_key)
    if layer not in profile.layers or profile.model_name != model_name:
        raise ValueError("Custom SAE source does not match the selected model/hook")
    sae, meta = load_sae_for_profile(sae_key, layer, device=device)
    config = getattr(sae, "cfg", None)
    if not all(callable(getattr(sae, method, None)) for method in ("encode", "decode", "forward")):
        raise ValueError("Original custom SAE contract requires encode/decode/forward")
    if (
        config is None or int(config.d_in) != MODEL_CONFIGS[model_name]["d_model"]
        or meta["hook_name"] != profile.hook_name(layer)
    ):
        raise ValueError("Wrong native SAE dimensions or hookpoint")
    llm_dtype = MODEL_CONFIGS[model_name]["dtype"]
    sae = sae.to(dtype=torch.float32 if llm_dtype == "float32" else torch.bfloat16)
    if not hasattr(sae, "W_dec"):
        raise ValueError("Native custom SAE must expose W_dec")
    source_sha = sha256_state_dict(sae)
    selected = [(f"opensae_custom_{sae_key}_L{layer}", sae)]
    # run_evals is from the ORIGINAL pinned SAEBench source, not copied math.
    run_evals(model_name=model_name, selected_saes=selected,
              llm_batch_size=MODEL_CONFIGS[model_name]["batch_size"],
              llm_dtype=llm_dtype, device=device, eval_types=[eval_type],
              api_key=None, force_rerun=True, save_activations=False)
    paths = sorted(Path("eval_results").rglob("*.json"))
    if not paths or len(paths) > 100:
        raise RuntimeError("Native runner returned without actual result JSON files")
    return {
        "status": "original_custom_sae_evaluator_completed",
        "source_model": model_name,
        "source_sae": sae_key, "sae_state_sha256": source_sha,
        "eval_type": eval_type, "source_hook": meta["hook_name"],
        "native_output_files": [
            {"path": str(x), "sha256": hashlib.sha256(x.read_bytes()).hexdigest()}
            for x in paths
        ],
    }


def run_custom(
    *, upstream: Path, upstream_commit: str, sae_key: str,
    layer: int, model_name: str, eval_type: str, device: str,
    output: Path, approved: bool = False, timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Isolated, time-bounded native run; source and trainer are immutable."""
    if not approved:
        raise PermissionError("Custom benchmark requires owner GPU/dataset approval")
    if not SHA.fullmatch(upstream_commit) or eval_type not in ALLOWED:
        raise ValueError("Supported native tool commit and evaluation required")
    if not 60 <= timeout_seconds <= 7200:
        raise ValueError("Bounded C1/C2 benchmark duration required")
    if output.exists():
        raise FileExistsError("Native custom benchmark records are immutable")
    actual = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        text=True, timeout=20,
    ).strip()
    if actual != upstream_commit:
        raise ValueError("Upstream custom benchmark source commit mismatch")
    output.mkdir(parents=True, mode=0o700)
    args = [sys.executable, "-m", "saebase.wp12.custom_saebench",
            "--worker", "--upstream", str(upstream),
            "--sae-key", sae_key, "--layer", str(layer),
            "--model-name", model_name, "--eval-type", eval_type,
            "--device", device]
    proc = subprocess.run(
        args, cwd=output, capture_output=True, text=True,
        timeout=timeout_seconds, check=False,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )
    (output / "native.stdout.log").write_text(proc.stdout)
    (output / "native.stderr.log").write_text(proc.stderr)
    record_file = output / "native-custom-summary.json"
    if proc.returncode or not record_file.is_file():
        raise RuntimeError("Original custom SAE evaluation failed; inspect private logs")
    record = json.loads(record_file.read_text())
    if record.get("status") != "original_custom_sae_evaluator_completed":
        raise RuntimeError("Native result is incomplete")
    return {**record, "original_saebench_commit": actual}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--sae-key", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--eval-type", choices=sorted(ALLOWED), required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = p.parse_args()
    if not args.worker:
        p.error("Use the approved run_custom() wrapper, not the worker entry directly")
    record = native_worker(
        upstream=args.upstream, sae_key=args.sae_key,
        layer=args.layer, model_name=args.model_name,
        eval_type=args.eval_type, device=args.device,
    )
    Path("native-custom-summary.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
