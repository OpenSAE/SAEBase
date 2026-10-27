#!/usr/bin/env python3
"""M0 real-weight reference: verified Pythia + SAE capture, patch and provenance.

The CLI uses published checkpoints (downloads are opt-in). Nothing in this
module claims real-checkpoint acceptance unless those weights are actually
loaded and its numerical checks succeed. The pure record_reference function
is intentionally testable with small deterministic model/SAE doubles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saebase.api import capture_features, evaluate_edit, load_registered
from saebase.datasets.SimpleCopying.prepare import prepare_for_model
from saebase.datasets.prompt_store import load_manifest, load_prompts
from saebase.sae.registry import get_profile

MODEL_KEY = "pythia70m"
SAE_KEY = "pythia70m_resid_sm"
LAYER = 3


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for buf in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def _sha_weights(module: Any) -> str:
    """Hash the full sorted state dict (not just a checkpoint name)."""
    if not callable(getattr(module, "state_dict", None)):
        raise TypeError("Cannot attest weights without state_dict()")
    state = module.state_dict()
    if not state:
        raise ValueError("Cannot attest an empty state_dict")
    sha = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State value {name!r} is not a tensor")
        a = value.detach().contiguous().cpu()
        sha.update(json.dumps([name, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
        sha.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def _versions() -> dict[str, str | None]:
    names = [
        "torch",
        "numpy",
        "omegaconf",
        "hydra-core",
        "transformer-lens",
        "sae-lens",
        "huggingface-hub",
        "safetensors",
        "transformers",
    ]
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _as_tokens(model: Any, example: Mapping[str, Any]) -> torch.Tensor:
    ids = model.tokens(prompt=str(example["prompt"]))
    recorded = example.get("tokenization", {}).get("input_ids")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("Reference prompt has no prepared input_ids")
    if ids[0].detach().cpu().tolist() != recorded:
        raise RuntimeError("Model tokenization differs from prepared prompt token IDs")
    return ids


def _probe(vector: torch.Tensor, *, n: int = 12) -> list[float]:
    return vector.detach().float().cpu().flatten()[:n].tolist()


@torch.no_grad()
def record_reference(
    bundle: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    reference_type: str = "synthetic_test",
) -> dict[str, Any]:
    """Record real operations and reject token/site/reconstruction drift.

    Called with a live bundle by main(); test doubles are marked synthetic_test.
    Tokens, activation probes, logits and a full model/SAE state fingerprint
    are persisted so future batch/store implementations have a reference.
    """
    if not prompts:
        raise ValueError("At least one prepared prompt is required")
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    model, sae, site = bundle.model, bundle.sae, bundle.site
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    identity = bundle.identity
    if identity.hook_name != site.hook_name or identity.layer != site.layer:
        raise ValueError("SAE identity and model hook differ")
    model_weights_sha = _sha_weights(model.model)
    sae_weights_sha = _sha_weights(sae)
    tokenizer = getattr(model.model, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer_record = {
        "class": (type(tokenizer).__module__ + "." + type(tokenizer).__qualname__)
        if tokenizer is not None
        else None,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode()).hexdigest()
            if isinstance(chat_template, str)
            else None
        ),
    }
    rows = []
    for sample in prompts:
        tokens = _as_tokens(model, sample)
        snapshot = capture_features(model, sae, site=site, tokens=tokens)
        model.assert_identity_patch(tokens=tokens, site=site, rtol=rtol, atol=atol)
        baseline = model.forward_logits(tokens=tokens)
        replay = model.patch_logits(tokens=tokens, site=site, activation=snapshot.activation)
        err = float((baseline.float() - replay.float()).abs().max().item())
        if not math.isfinite(err):
            raise RuntimeError("Identity replay produced non-finite logits")
        codes = snapshot.codes
        top = int(torch.argmax(codes.abs()).item())
        if float(codes[top].abs()) <= 0:
            raise RuntimeError("SAE did not activate any feature for the reference prompt")
        changed = evaluate_edit(
            model,
            sae,
            site=site,
            tokens=tokens,
            replacements={top: 0.0},
            verify_identity=True,
        )
        if not bool(torch.isfinite(changed.logit_delta).all()):
            raise RuntimeError("Non-finite edited logit difference")
        row = {
            "sample_id": str(sample.get("id", "missing-id")),
            "prompt_sha256": hashlib.sha256(str(sample["prompt"]).encode()).hexdigest(),
            "token_ids": tokens[0].detach().cpu().tolist(),
            "token_count": int(tokens.shape[1]),
            "site": {"hook": site.hook_name, "layer": site.layer, "position": site.position},
            "activation_dim": int(snapshot.activation.numel()),
            "sae_width": int(codes.numel()),
            "activation_l2": float(snapshot.activation.float().norm().item()),
            "activation_probe": _probe(snapshot.activation),
            "codes_l0": int(torch.count_nonzero(codes).item()),
            "selected_feature": top,
            "selected_feature_value": float(codes[top].item()),
            "reconstruction_mse": float(
                ((snapshot.activation - snapshot.reconstruction).float() ** 2).mean().item()
            ),
            "baseline_logit_probe": _probe(baseline),
            "identity_max_logit_abs_error": err,
            "edit_logit_delta_l2": float(changed.logit_delta.float().norm().item()),
            "edited_logit_probe": _probe(changed.edited_logits),
        }
        if not math.isfinite(row["reconstruction_mse"]):
            raise RuntimeError("Reconstruction MSE is not finite")
        rows.append(row)
    report = {
        "schema": "saebase.m0.reference.v1",
        "status": "passed",
        "reference_type": reference_type,
        "real_checkpoint_validated": reference_type == "real_pythia70m_sae",
        "model_key": identity.model_key,
        "sae_key": identity.sae_key,
        "site": {"hook": site.hook_name, "layer": site.layer, "position": site.position},
        "identity": identity.to_dict(),
        "model_state_sha256": model_weights_sha,
        "sae_state_sha256": sae_weights_sha,
        "tokenizer": tokenizer_record,
        "seed": int(seed),
        "tolerances": {"rtol": rtol, "atol": atol},
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "sample_count": len(rows),
        "max_identity_error": max(row["identity_max_logit_abs_error"] for row in rows),
        "samples": rows,
    }
    json.dumps(report, allow_nan=False)
    return report


def compare_references(current: Mapping[str, Any], prior: Mapping[str, Any]) -> None:
    """Strict identity match; numeric probes compared using reference tolerances."""
    if prior.get("real_checkpoint_validated") != current.get("real_checkpoint_validated"):
        raise ValueError("Reference provenance type differs")
    for name in (
        "model_key",
        "sae_key",
        "model_state_sha256",
        "sae_state_sha256",
        "tokenizer",
        "site",
        "seed",
    ):
        if current.get(name) != prior.get(name):
            raise RuntimeError(f"Reference mismatch: {name}")
    ref_rows, rows = prior.get("samples", []), current.get("samples", [])
    if len(rows) != len(ref_rows):
        raise RuntimeError("Reference sample count differs")
    rtol = float(current["tolerances"]["rtol"])
    atol = float(current["tolerances"]["atol"])
    for actual, ref in zip(rows, ref_rows):
        for key in (
            "sample_id",
            "prompt_sha256",
            "token_ids",
            "selected_feature",
            "activation_dim",
            "sae_width",
            "codes_l0",
        ):
            if actual[key] != ref[key]:
                raise RuntimeError(f"Reference mismatch for {ref['sample_id']}: {key}")
        for key in (
            "activation_probe",
            "baseline_logit_probe",
            "edited_logit_probe",
            "selected_feature_value",
            "reconstruction_mse",
            "edit_logit_delta_l2",
        ):
            if not np.allclose(actual[key], ref[key], rtol=rtol, atol=atol):
                raise RuntimeError(f"Reference numerical drift for {ref['sample_id']}: {key}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0 real checkpoint reference capture")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-generate", type=int, default=120)
    parser.add_argument("--min-kept", type=int, default=5)
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=Path("m0-artifacts/data"))
    parser.add_argument("--output", type=Path, default=Path("m0-artifacts/m0_reference.json"))
    parser.add_argument("--reuse-prompts", action="store_true")
    parser.add_argument("--compare-to", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.n_prompts < 1 or args.min_kept < args.n_prompts or args.n_generate < args.min_kept:
        parser.error("require 1 <= n-prompts <= min-kept <= n-generate")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from saebase.runtime import verify_runtime

    verify_runtime(context="M0 real checkpoint acceptance", mode="analysis")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bundle = load_registered(
        model_key=MODEL_KEY,
        sae_key=SAE_KEY,
        layer=LAYER,
        device=args.device,
    )
    if bundle.identity.model_key != MODEL_KEY or bundle.identity.sae_key != SAE_KEY:
        raise RuntimeError("Loaded weights do not match requested Pythia profile")
    args.data_dir = args.data_dir.expanduser().resolve()
    if not args.reuse_prompts:
        prepare_for_model(
            MODEL_KEY,
            bundle.model.model,
            model_name=get_profile(SAE_KEY).model_name,
            loaded_model_name=bundle.identity.loaded_model_name or MODEL_KEY,
            n_generate=args.n_generate,
            min_kept=args.min_kept,
            seed=args.seed,
            data_root=args.data_dir,
        )
    manifest = load_manifest("SimpleCopying", MODEL_KEY, data_root=args.data_dir)
    if not args.reuse_prompts and int(manifest["seed"]) != args.seed:
        raise RuntimeError("Prepared manifest seed differs from requested reference seed")
    prompts = load_prompts("SimpleCopying", MODEL_KEY, data_root=args.data_dir)
    if len(prompts) < args.n_prompts:
        raise RuntimeError("Not enough prepared prompts for requested reference")
    report = record_reference(
        bundle,
        prompts[: args.n_prompts],
        seed=args.seed,
        reference_type="real_pythia70m_sae",
    )
    from saebase.datasets.artifact_layout import manifest_path, prompts_path

    report["prompt_manifest"] = {
        "behavior": manifest["behavior"],
        "model_key": manifest["model_key"],
        "kept_count": manifest["kept_count"],
        "manifest_sha256": _hash_file(
            manifest_path("SimpleCopying", MODEL_KEY, data_root=args.data_dir)
        ),
        "prompts_sha256": _hash_file(
            prompts_path("SimpleCopying", MODEL_KEY, data_root=args.data_dir)
        ),
    }
    if args.compare_to:
        previous = json.loads(args.compare_to.read_text(encoding="utf-8"))
        compare_references(report, previous)
        report["reference_comparison"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[M0] REAL Pythia + SAE reference passed, samples={len(report['samples'])}")
    print(f"[M0] Model sha256: {report['model_state_sha256']}")
    print(f"[M0] SAE sha256:   {report['sae_state_sha256']}")
    print(f"[M0] Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
