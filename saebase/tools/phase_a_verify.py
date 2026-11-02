"""Real-weight acceptance for SAEBase Phase A model/SAE coverage.

Run one entry at a time. No synthetic fixtures count as verified checkpoints.
Requires the original model and SAE weights, and often a CUDA GPU / HF access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
from saebase.api.sae_loading import load_model_and_sae
from saebase.api.sae_profiles import get_profile
from saebase.store import sha256_state_dict


TARGETS = {
    "gemma270m": ("gemma270m_scope_resid", 5, 640),
    "gemma1b": ("gemma1b_scope_resid", 13, 1152),
    "gemma2_2b": ("gemma2_2b_pt_l25_canonical", 25, 2304),
    "qwen35_2b": ("qwen35_2b_resid_w32k", 12, 2048),
    "qwen3_8b": ("qwen3_8b_resid_w64k", 18, 4096),
    "llama31_8b": ("llama31_8b_lxr_8x", 16, 4096),
}

PROMPTS = (
    "The capital of France is",
    "A scientist explained the results carefully.",
    "Small birds fly across the morning sky.",
)


def identity_patch(model: ModelAdapter, snapshot, site: ActivationSite) -> float:
    """Patching the *same* saved activation must preserve original logits."""
    baseline = model.forward_logits(tokens=snapshot.tokens).detach().float().cpu()
    patched = (
        model.patch_logits(tokens=snapshot.tokens, site=site, activation=snapshot.activation)
        .detach()
        .float()
        .cpu()
    )
    if baseline.shape != patched.shape:
        raise ValueError(f"identity logits shape mismatch: {baseline.shape} vs {patched.shape}")
    return float((baseline - patched).abs().max().item())


def verify_target(name: str, *, device: str = "cuda", prompts=PROMPTS) -> dict:
    sae_key, layer, expected_width = TARGETS[name]
    profile = get_profile(sae_key)
    if profile.model_key not in {
        "gemma270m",
        "gemma1b",
        "gemma2_2b_base",
        "qwen35_2b",
        "qwen3_8b",
        "llama31_8b",
    }:
        raise ValueError("phase A profile unexpectedly points to a different model")
    raw_model, sae, metadata = load_model_and_sae(sae_key, layer, device=device)
    model = ModelAdapter(raw_model)
    site = ActivationSite(layer, profile.hook_name(layer), position=-1)
    hook = metadata["hook_name"]
    if hook != site.hook_name:
        raise ValueError(f"SAE actual hook {hook!r} disagrees with model site {site.hook_name!r}")
    if int(sae.cfg.d_in) != expected_width:
        raise ValueError(f"SAE dimension {sae.cfg.d_in} != expected {expected_width}")
    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise ValueError("SAE must expose encode/decode")
    model.eval() if callable(getattr(model, "eval", None)) else None
    results = []
    for prompt in prompts:
        snap = capture_features(model, sae, site=site, prompt=prompt)
        x = snap.activation.detach().float().cpu()
        z = snap.codes.detach().float().cpu()
        recon = snap.reconstruction.detach().float().cpu()
        if x.shape != (expected_width,) or z.ndim != 1 or z.numel() != sae.cfg.d_sae:
            raise ValueError("activation/SAE dimensions disagree")
        if recon.shape != x.shape or not torch.isfinite(x).all() or not torch.isfinite(z).all():
            raise ValueError("invalid SAE activations or reconstruction")
        if not torch.isfinite(recon).all():
            raise ValueError("nonfinite SAE reconstruction")
        err = identity_patch(model, snap, site)
        if not err <= 1e-2:
            raise ValueError(f"identity patch maximum logit error {err} > 0.01")
        # One genuine SAE feature ablation must also propagate through the
        # captured model hook and change the downstream logits.
        if not torch.count_nonzero(z):
            raise ValueError("SAE has no active features on acceptance prompt")
        selected = int(torch.argmax(torch.abs(z)).item())
        edited = edit_features(sae, snap, replacements={selected: 0.0})
        base_logits = model.forward_logits(tokens=snap.tokens).detach().float()
        edited_logits = (
            model.patch_logits(tokens=snap.tokens, site=site, activation=edited.activation)
            .detach()
            .float()
        )
        feature_logit_delta_l2 = float(torch.linalg.vector_norm(base_logits - edited_logits).item())
        if (
            not torch.isfinite(torch.tensor(feature_logit_delta_l2))
            or feature_logit_delta_l2 <= 1e-8
        ):
            raise ValueError("active SAE feature ablation did not measurably affect logits")
        results.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "l0": int(torch.count_nonzero(z).item()),
                "reconstruction_mse": float(torch.mean((x - recon) ** 2).item()),
                "identity_patch_max_abs_logit_error": err,
                "ablated_feature_id": selected,
                "feature_ablation_logit_delta_l2": feature_logit_delta_l2,
            }
        )
    return {
        "schema": "saebase.phase_a.real_acceptance.v1",
        "model": name,
        "model_key": profile.model_key,
        "sae_key": sae_key,
        "model_repo": profile.model_name,
        "sae_repo": profile.repo_id or profile.sae_release,
        "sae_id": profile.sae_id(layer),
        "layer": layer,
        "hook": site.hook_name,
        "loaded_model_name": metadata.get("loaded_model_name"),
        "model_state_sha256": sha256_state_dict(raw_model),
        "sae_state_sha256": sha256_state_dict(sae),
        "d_model": expected_width,
        "d_sae": int(sae.cfg.d_sae),
        "device": device,
        "sample_count": len(results),
        "probes": results,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*TARGETS, "all"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("phase-a-results"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(TARGETS, indent=2))
        return 0
    if args.model is None:
        parser.error("--model is required unless --list is set")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = list(TARGETS) if args.model == "all" else [args.model]
    complete = []
    failures = 0
    for name in selected:
        # A real checkpoint run must produce evidence even when unavailable.
        # Never count a failed, missing, or memory-killed job as verified.
        try:
            result = verify_target(name, device=args.device)
        except Exception as exc:
            failures += 1
            message = str(exc)
            token = os.environ.get("HF_TOKEN")
            if token:
                message = message.replace(token, "[redacted]")
            result = {
                "schema": "saebase.phase_a.real_acceptance.v1",
                "status": "blocked"
                if isinstance(exc, (OSError, ImportError, RuntimeError))
                else "failed",
                "model": name,
                "device": args.device,
                "error_type": type(exc).__name__,
                "error": message[:1500],
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "real_weights_validated": False,
            }
        target = args.output / f"{name}.json"
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        complete.append({"model": name, "status": result["status"], "evidence": str(target)})
        print(f"{result['status'].upper()}: {name}: {target}", flush=True)
        if result["status"] != "passed":
            print(
                json.dumps(
                    {"error_type": result["error_type"], "error": result["error"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(json.dumps(complete, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
