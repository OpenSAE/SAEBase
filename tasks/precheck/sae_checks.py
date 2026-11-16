"""Select SAE sites, check actual encode/decode and measure real model SAE pair behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from models.hf_hooks_backend import load_hf_hooks_model
from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
from saebase.api.sae_loading import load_model_and_sae, load_model_for_profile, load_sae_for_profile
from saebase.api.sae_profiles import get_profile
from saebase.store import sha256_state_dict
from tasks.benchmark.sites.catalog import load_coverage


def select_sites(
    model_cfg: Any, selection: Any, *, layer: int | None = None
) -> list[dict[str, Any]]:
    """Model-owned lists, never a second global SAE Registry.

    A free-text SAE key must still refer to a Model-owned profile. The legacy
    profile table is used only to resolve current checkpoint and hook parameters.
    """
    from tasks.sae_runtime import resolve_sae

    available = list(model_cfg.get("sae_profiles", []))
    defaults = list(model_cfg.get("default_saes", []))
    if selection in ("default", None):
        chosen = defaults
    elif selection == "all":
        chosen = available
    elif isinstance(selection, str):
        chosen = [selection]
    else:
        chosen = list(selection)
    if not chosen:
        raise ValueError("Model has no default SAE; provide model.default_saes or task.sae_list")
    rows = []
    seen = set()
    for entry in chosen:
        key = entry.get("sae", entry.get("key")) if hasattr(entry, "get") else entry
        requested = entry.get("layer", layer) if hasattr(entry, "get") else layer
        result = resolve_sae(model_cfg, key, requested, operation="precheck")
        identity = (result.sae_key, result.layer)
        if identity in seen:
            raise ValueError(f"Duplicate SAE check {identity}")
        seen.add(identity)
        rows.append({"sae_key": result.sae_key, "layer": result.layer, "hook": result.hook})
    return rows


def check_real_sae(
    *,
    model_key: str,
    sae_key: str,
    layer: int,
    prompt: str,
    device: str,
    operation: str = "precheck",
    annotation_options: dict | None = None,
    capture_mode: str = "non_whiten",
    whiten_chunk_size: int = 256,
) -> dict[str, Any]:
    """Use real checkpoints and real tokens; neither fake activations nor random smoke."""
    import torch
    from omegaconf import OmegaConf

    from saebase.api import capture_features
    from tasks.sae_runtime import load_selection, resolve_sae

    if not prompt or not prompt.strip():
        raise ValueError("Precheck requires a nonempty real prompt for SAE capture")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_RESOURCE: no CUDA device available")
    selection = resolve_sae(
        OmegaConf.create({"name": model_key}), sae_key, layer, operation=operation
    )
    pair = load_selection(selection, device)
    if capture_mode == "whiten":
        from saebase.api.features import encode_activations, snapshot_from_activation

        tokens = pair.model.tokens(prompt=prompt)
        sequence = pair.model.capture_sequence(tokens=tokens, site=pair.site)
        for start in range(0, sequence.shape[0], whiten_chunk_size):
            encode_activations(pair.sae, sequence[start : start + whiten_chunk_size])
        snapshot = snapshot_from_activation(
            pair.sae, tokens=tokens, site=pair.site, activation=sequence[-1]
        )
    else:
        snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
    h, z, rebuilt = snapshot.activation, snapshot.codes, snapshot.reconstruction
    if h.ndim != 1 or z.ndim != 1 or rebuilt.shape != h.shape:
        raise RuntimeError("SAE capture/encode/decode returned invalid shapes")
    if not all(bool(torch.isfinite(x).all()) for x in (h, z, rebuilt)):
        raise RuntimeError("SAE activation, latent, or reconstruction is non-finite")
    if h.numel() != pair.identity.d_in or z.numel() != pair.identity.d_sae:
        raise RuntimeError("SAE checkpoint dimensions do not match the captured activation")
    report = {
        "status": "PASS",
        "scope": "single real prompt / single SAE site; not scientific SAE quality acceptance",
        "model_key": model_key,
        "capture_mode": capture_mode,
        "sae_key": sae_key,
        "layer": layer,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "identity": pair.identity.to_dict(),
        "activation_shape": list(h.shape),
        "latent_shape": list(z.shape),
        "l0": int(torch.count_nonzero(z).item()),
        "reconstruction_mse": float(torch.mean((h.float() - rebuilt.float()) ** 2).item()),
    }

    if annotation_options is not None:
        from saebase.api.neuronpedia import annotate_features
        from tasks.precheck.report import blocked

        annotation, _ = annotate_features(pair, [], annotation_options)
        report["annotation_source"] = annotation
        report["annotation_feature_coverage"] = "not_assessed; capture records each feature lookup"
        if annotation["status"] == "blocked":
            return {
                **report,
                **blocked(
                    stage="neuronpedia_source",
                    code="BLOCKED_ANNOTATION_SOURCE",
                    reason=annotation["reason"],
                    attempted=["verify exact Neuronpedia source and export cache"],
                    repair="Prepare the declared source exports or correct the recorded service/cache failure",
                ),
            }
    return report


def load_cpu_pair(sae_key: str, *, device: str = "cpu"):
    """Return one real CPU ModelAdapter + SAE using the registered native site."""
    if device != "cpu":
        raise ValueError("Stage B/C acceptance is intentionally CPU-only")
    entry = load_coverage(CATALOG).find(sae_key)
    if not entry.model_pair_cpu_candidate:
        raise ValueError("model requires Phase D hardware or access")
    layer = entry.cpu_probe_layers[0]
    p = get_profile(sae_key)
    if p.model_key == "r1_distill_qwen15b":
        # Native HF decoder hook is the unprocessed resid_post used to train
        # the reasoning SAE. Avoid a duplicate HF -> TL model conversion.
        handle = load_hf_hooks_model(
            p.model_name,
            key=p.model_key,
            family=p.base.family,
            device="cpu",
            dtype=torch.bfloat16,
        )
        model_raw = handle.model
        model_name = handle.loaded_name
    else:
        model_raw, model_name = load_model_for_profile(sae_key, device="cpu")
    model = ModelAdapter(model_raw)
    sae, meta = load_sae_for_profile(sae_key, layer, device="cpu")
    site = ActivationSite(layer, p.hook_name(layer), position=-1)
    if meta["hook_name"] != site.hook_name:
        raise ValueError("SAE native checkpoint hook mismatch")
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    return entry, p, model, sae, meta, site, model_name, model_raw


def _check_protocol(name):
    from conf import load_group

    return load_group("benchmark", name).benchmark


def _reference_targets():
    from importlib.resources import files

    from omegaconf import OmegaConf

    raw = OmegaConf.load(files("conf") / "benchmark/catalog.yaml").reference_pairs
    return {name: (p.sae_key, int(p.layer), int(p.width)) for name, p in raw.items()}


TARGETS = _reference_targets()
PROMPTS = tuple(_check_protocol("reference_pair").prompts)
CPU_PROMPTS = tuple(_check_protocol("cpu_pair").prompts)
CATALOG = Path(__file__).resolve().parents[2] / "conf/coverage/stage_bc.json"


def measure_pair(model, sae, site, prompts, *, width=None, tolerance=0.01):
    """Observe real capture, shape, identity replay and active-feature removal."""
    results = []
    width = int(sae.cfg.d_in) if width is None else width
    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise ValueError("SAE requires encode and decode operations")
    for prompt in prompts:
        snap = capture_features(model, sae, site=site, prompt=prompt)
        x, z, recon = [
            v.detach().float().cpu() for v in (snap.activation, snap.codes, snap.reconstruction)
        ]
        if x.shape != (width,) or z.shape != (int(sae.cfg.d_sae),) or recon.shape != x.shape:
            raise ValueError("activation/SAE dimensions disagree")
        if not all(bool(torch.isfinite(v).all()) for v in (x, z, recon)):
            raise ValueError("nonfinite activation, codes or reconstruction")
        base = model.forward_logits(tokens=snap.tokens).detach().float()
        replay = (
            model.patch_logits(tokens=snap.tokens, site=site, activation=snap.activation)
            .detach()
            .float()
        )
        if base.shape != replay.shape:
            raise ValueError("identity logits shape mismatch")
        error = float((base - replay).abs().max().item())
        if not 0 <= error <= tolerance:
            raise ValueError(f"identity patch maximum logit error {error} > {tolerance}")
        if not torch.count_nonzero(z):
            raise ValueError("SAE has no active features on the acceptance prompt")
        selected = int(torch.argmax(z.abs()).item())
        edited = edit_features(sae, snap, replacements={selected: 0.0})
        changed = (
            model.patch_logits(tokens=snap.tokens, site=site, activation=edited.activation)
            .detach()
            .float()
        )
        delta = float(torch.linalg.vector_norm(base - changed).item())
        if not torch.isfinite(torch.tensor(delta)) or delta <= 1e-8:
            raise ValueError("active SAE feature ablation did not measurably affect logits")
        results.append(
            dict(
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                l0=int(torch.count_nonzero(z).item()),
                reconstruction_mse=float(torch.mean((x - recon) ** 2).item()),
                identity_patch_max_abs_logit_error=error,
                ablated_feature_id=selected,
                feature_ablation_logit_delta_l2=delta,
            )
        )
    return results


def verify_pair(sae_key, *, device="cpu"):
    entry, profile, model, sae, meta, site, model_name, raw = load_cpu_pair(sae_key, device=device)
    rows = measure_pair(model, sae, site, CPU_PROMPTS)
    observations = [
        dict(
            prompt_sha256=x["prompt_sha256"],
            l0=x["l0"],
            reconstruction_mse=x["reconstruction_mse"],
            identity_max_abs_logit_error=x["identity_patch_max_abs_logit_error"],
            ablated_feature=x["ablated_feature_id"],
            feature_delta_logit_l2=x["feature_ablation_logit_delta_l2"],
        )
        for x in rows
    ]
    return dict(
        schema="saebase.coverage.cpu_pair_acceptance.v1",
        status="model_and_sae_verified",
        sae_key=sae_key,
        model_key=entry.model_key,
        model_repo=model_name,
        sae_repo=profile.repo_id or profile.sae_release,
        layer=site.layer,
        hook=site.hook_name,
        model_state_sha256=sha256_state_dict(raw),
        sae_state_sha256=sha256_state_dict(sae),
        sample_count=len(rows),
        observations=observations,
        device=device,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def verify_target(name, *, device="cuda", prompts=PROMPTS):
    sae_key, layer, width = TARGETS[name]
    profile = get_profile(sae_key)
    raw, sae, meta = load_model_and_sae(sae_key, layer, device=device)
    model = ModelAdapter(raw)
    site = ActivationSite(layer, profile.hook_name(layer), position=-1)
    if meta["hook_name"] != site.hook_name or int(sae.cfg.d_in) != width:
        raise ValueError("SAE source hook or dimension disagrees with requested model site")
    probes = measure_pair(model, sae, site, prompts, width=width)
    return dict(
        schema="saebase.phase_a.real_acceptance.v1",
        model=name,
        model_key=profile.model_key,
        sae_key=sae_key,
        model_repo=profile.model_name,
        sae_repo=profile.repo_id or profile.sae_release,
        sae_id=profile.sae_id(layer),
        layer=layer,
        hook=site.hook_name,
        loaded_model_name=meta.get("loaded_model_name"),
        model_state_sha256=sha256_state_dict(raw),
        sae_state_sha256=sha256_state_dict(sae),
        d_model=width,
        d_sae=int(sae.cfg.d_sae),
        device=device,
        sample_count=len(probes),
        probes=probes,
        python=platform.python_version(),
        torch=torch.__version__,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        status="passed",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Check selected real model and SAE checkpoint pairs")
    selected = p.add_mutually_exclusive_group()
    selected.add_argument("--model", choices=(*TARGETS, "all"))
    selected.add_argument("--sae-key")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--output", type=Path, default=Path("pair-results"))
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)
    if args.list:
        print(json.dumps(TARGETS, indent=2))
        return 0
    if not args.model and not args.sae_key:
        p.error("select --model or --sae-key")
    args.output.mkdir(parents=True, exist_ok=True)
    names = (
        [args.sae_key] if args.sae_key else list(TARGETS) if args.model == "all" else [args.model]
    )
    failed = False
    for name in names:
        try:
            result = (
                verify_pair(name, device=args.device or "cpu")
                if args.sae_key
                else verify_target(name, device=args.device or "cuda")
            )
        except Exception as exc:
            from tasks.precheck.report import error_result

            detail = error_result(
                exc,
                stage=f"model_sae_pair:{name}",
                attempted=[
                    "load requested weights",
                    "capture",
                    "identity replay",
                    "feature ablation",
                ],
            )
            message = str(exc)
            if os.getenv("HF_TOKEN"):
                message = message.replace(os.environ["HF_TOKEN"], "[redacted]")
            detail["reason"] = message[:1500]
            detail.pop("traceback", None)
            result = {
                "schema": "saebase.coverage.cpu_pair_acceptance.v1"
                if args.sae_key
                else "saebase.phase_a.real_acceptance.v1",
                "sae_key": args.sae_key if args.sae_key else TARGETS[name][0],
                "device": args.device or ("cpu" if args.sae_key else "cuda"),
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                **detail,
                "status": "blocked" if detail["status"] == "BLOCKED" else "failed",
                "error_type": type(exc).__name__,
                "error": message[:1500],
                "model": name,
                "real_weights_validated": False,
            }
            failed = True
        name_out = f"{name}_pair.json" if args.sae_key else f"{name}.json"
        (args.output / name_out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
