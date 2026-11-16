"""Purpose: check prepared inputs, measure SAE activation health and summarize observed reconstructions.

Scope: the registered activation-health evaluation and its per-prompt result readers.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from saebase.datasets.artifacts import assert_available, canonicalize_behavior, load_manifest
from saebase.datasets.legacy_prompts import load_behavior_prompts

from models.hooked_lm_common import print_vram_gb
from saebase.api.sae_contracts import (
    assert_activation_sae_shapes,
    assert_finite_tensors,
    assert_manifest_matches,
    assert_prompt_record,
    latent_l0,
    reconstruction_stats,
)
from saebase.api.sae_loading import load_model_and_sae
from saebase.api.sae_profiles import get_profile
from saebase.registry import (
    get_dataset_plugin,
    get_model_plugin,
    get_prompt_source_plugin,
    get_task_plugin,
)
from saebase.results import EvaluationResult


@torch.no_grad()
def analyze_prompt_activation(
    *,
    model: Any,
    sae: Any,
    hook_name: str,
    ex: dict[str, Any],
    prompt_idx: int,
) -> dict[str, Any]:
    """Run one prompt through model hook + SAE; return health metrics."""
    ctx = f"prompt_idx={prompt_idx}"
    assert_prompt_record(ex, model=model, ctx=ctx)

    prompt = str(ex["prompt"])
    tok = ex.get("tokenization") if isinstance(ex.get("tokenization"), dict) else {}
    if isinstance(tok.get("input_ids"), list) and tok["input_ids"]:
        tokens = torch.tensor([tok["input_ids"]], device=next(model.parameters()).device)
    else:
        tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    if hook_name not in cache:
        raise RuntimeError(f"{ctx}: hook {hook_name!r} missing from cache")

    h = cache[hook_name][0, -1].float()
    z = sae.encode(h.unsqueeze(0))[0].float()
    h_hat = sae.decode(z.unsqueeze(0))[0].float()

    assert_activation_sae_shapes(h, z, sae, ctx=ctx)
    assert_finite_tensors(h, z, h_hat, ctx=ctx)
    l0 = latent_l0(z)
    if l0 <= 0:
        raise RuntimeError(f"{ctx}: degenerate SAE latent L0={l0}")

    recon = reconstruction_stats(h, h_hat)
    return {
        "prompt_idx": prompt_idx,
        "prompt_id": ex.get("id", f"idx-{prompt_idx}"),
        "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "token_count": int(tokens.shape[1]),
        "activation_shape": list(h.shape),
        "activation_norm": float(h.norm().item()),
        "d_in": int(sae.cfg.d_in),
        "latent_shape": list(z.shape),
        "d_sae": int(sae.cfg.d_sae),
        "l0": l0,
        **recon,
        "ok": True,
    }


def write_run_summary(payload: dict[str, Any], path: Path | None = None) -> Path:
    out = path or Path.cwd() / "sae_activation_summary.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oks = [r for r in rows if r.get("ok")]
    fails = [r for r in rows if not r.get("ok")]
    l0s = sorted(float(r["l0"]) for r in oks)
    mses = sorted(float(r["reconstruction_mse"]) for r in oks)
    fves = sorted(float(r["fve"]) for r in oks if "fve" in r)

    def _median(vals: list[float]) -> float | None:
        if not vals:
            return None
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return 0.5 * (vals[mid - 1] + vals[mid])

    return {
        "n_ok": len(oks),
        "n_fail": len(fails),
        "l0_median": _median(l0s),
        "reconstruction_mse_median": _median(mses),
        "fve_median": _median(fves) if fves else None,
        "fve_mean": (sum(fves) / len(fves)) if fves else None,
        "failures": [{"prompt_idx": r.get("prompt_idx"), "error": r.get("error")} for r in fails],
    }


def _prompt_source_name(cfg: DictConfig) -> str:
    if "prompt_source" in cfg and "name" in cfg.prompt_source:
        return str(cfg.prompt_source.name)
    return str(cfg.dataset.name)


def run_preflight(cfg: DictConfig) -> dict:
    """Validate registry gates + prompt artifact identity."""
    if str(cfg.task.name) != "sae":
        get_task_plugin(str(cfg.task.name))
    ps_name = _prompt_source_name(cfg)
    availability_check = assert_available
    manifest_loader = load_manifest
    if str(cfg.task.name) == "sae" or cfg.task.get("evaluation") == "activation_health":
        # The standalone compatibility CLI uses the existing PromptSource
        # contract; canonical Dataset plugins accept the current Task names.
        if str(cfg.task.name) == "sae":
            from copy import deepcopy

            current = deepcopy(cfg)
            current.task.name = "benchmark"
        else:
            current = cfg
        get_prompt_source_plugin(ps_name).assert_compatible(current)
        from saebase.datasets.artifacts import assert_available as legacy_available
        from saebase.datasets.artifacts import load_manifest as legacy_manifest

        availability_check = legacy_available
        manifest_loader = legacy_manifest
    else:
        get_dataset_plugin(ps_name).assert_task_supported(cfg)
    get_model_plugin(str(cfg.model.name)).assert_compatible(cfg)
    if "sae" not in cfg or "name" not in cfg.sae:
        raise ValueError("Select an explicit SAE profile")
    sae_key = str(cfg.sae.name)
    if sae_key == "none":
        raise ValueError("An actual SAE profile is required")

    behavior = canonicalize_behavior(ps_name)
    model_key = str(cfg.model.name)
    availability_check(behavior, model_key)

    profile = get_profile(sae_key)
    if model_key not in profile.allowed_model_keys():
        raise RuntimeError(
            f"sae.name={sae_key!r} allows models {sorted(profile.allowed_model_keys())}, "
            f"but model.name={model_key!r}"
        )

    require = True
    if "evaluation" in cfg and "require_prepared_prompts" in cfg.evaluation:
        require = bool(cfg.evaluation.require_prepared_prompts)
    elif "sae" in cfg and "require_prepared_prompts" in cfg.sae:
        require = bool(cfg.sae.require_prepared_prompts)

    if not require:
        raise ValueError(
            "activation_health in Phase A requires prepared prompts and a manifest; "
            "run datasets/prepare.py first"
        )

    data_root = cfg.paths.get("data_dir") if "paths" in cfg else None
    manifest = manifest_loader(
        behavior,
        model_key,
        data_root=str(data_root) if data_root is not None else None,
    )
    assert_manifest_matches(manifest, behavior=behavior, model_key=model_key)

    layer = None
    if "sae" in cfg:
        layer = cfg.sae.get("layer")
    if layer is None and "evaluation" in cfg:
        layer = cfg.evaluation.get("layer")
    from tasks.sae_runtime import resolve_sae

    selected = resolve_sae(cfg.model, sae_key, layer, operation="precheck")
    layer = selected.layer

    if model_key == "pythia410m" and profile.hook_kind != "hook_mlp_out":
        raise RuntimeError("pythia410m must use hook_mlp_out SAE sites")

    return {
        "behavior": behavior,
        "model_key": model_key,
        "sae_key": profile.key,
        "layer": layer,
        "manifest": manifest,
        "profile": profile,
        "hook_name": profile.hook_name(layer),
        "evaluation": str(cfg.task.get("evaluation", "activation_health")),
    }


def run_activation_health(cfg: DictConfig) -> EvaluationResult:
    meta = run_preflight(cfg)
    behavior = meta["behavior"]
    model_key = meta["model_key"]
    sae_key = meta["sae_key"]
    layer = int(meta["layer"])
    profile = meta["profile"]

    n = int(cfg.evaluation.get("n_prompts", cfg.sae.get("n_prompts", 5)))
    device = str(cfg.evaluation.get("device", cfg.sae.get("device", "cuda")))
    data_root = cfg.paths.get("data_dir") if "paths" in cfg else None
    if data_root is not None:
        data_root = str(data_root)

    manifest, prompts = load_behavior_prompts(behavior, model_key, n=n, data_root=data_root)

    model, sae, sae_meta = load_model_and_sae(
        sae_key, layer, device=device, model_key_override=model_key
    )
    hook_name = str(sae_meta["hook_name"])
    if model_key == "pythia410m" and "hook_mlp_out" not in hook_name:
        raise RuntimeError(f"pythia410m unexpected hook_name={hook_name!r}; expected hook_mlp_out")

    per_prompt: list[dict[str, Any]] = []
    for i, ex in enumerate(prompts):
        try:
            row = analyze_prompt_activation(
                model=model,
                sae=sae,
                hook_name=hook_name,
                ex=ex,
                prompt_idx=i,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "prompt_idx": i,
                "prompt_id": ex.get("id", f"idx-{i}"),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        per_prompt.append(row)
        if row.get("ok"):
            print(
                f"  [OK] #{i} L0={row['l0']} "
                f"||h||={row['activation_norm']:.3f} "
                f"mse={row['reconstruction_mse']:.6f}"
            )
        else:
            print(
                f"  [FAIL] #{i} model={model_key} behavior={behavior} "
                f"layer={layer} hook={hook_name} sae_id={sae_meta['sae_id']} "
                f"reason={row.get('error')}"
            )

    summary = summarize_rows(per_prompt)
    payload = {
        "task": "benchmark",
        "evaluation": "activation_health",
        "behavior": behavior,
        "model_key": model_key,
        "sae_key": sae_key,
        "loaded_model_name": sae_meta.get("loaded_model_name"),
        "layer": layer,
        "sae": {
            "release": sae_meta["sae_release"],
            "sae_id": sae_meta["sae_id"],
            "hook_name": hook_name,
            "d_in": sae_meta["d_in"],
            "d_sae": sae_meta["d_sae"],
            "loader": sae_meta["loader"],
        },
        "manifest": {
            "kept_count": manifest.get("kept_count"),
            "loaded_model_name": manifest.get("loaded_model_name"),
            "filter_policy": manifest.get("filter_policy"),
        },
        "hydra": OmegaConf.to_container(cfg, resolve=True),
        "summary": summary,
        "per_prompt": per_prompt,
        "profile_layers": list(profile.layers),
    }

    write_summary = bool(cfg.task.get("facets", {}).get("post_run_summary", True))
    if write_summary:
        out = write_run_summary(payload)
        print(f"[sae] wrote summary: {out}")

    print(
        f"[sae] {model_key}/{sae_key} L{layer} {behavior}: "
        f"ok={summary['n_ok']} fail={summary['n_fail']} "
        f"L0_median={summary['l0_median']} "
        f"mse_median={summary['reconstruction_mse_median']}"
    )
    print_vram_gb()

    del model, sae
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    ok = summary["n_ok"] > 0 and summary["n_fail"] == 0
    return EvaluationResult(
        ok=ok,
        task="benchmark",
        evaluation="activation_health",
        model_key=model_key,
        prompt_source=behavior,
        sae_key=sae_key,
        layer=layer,
        summary=summary,
        per_prompt=per_prompt,
        meta=payload,
        error=None if ok else f"Activation health failures: {summary['n_fail']}",
    )
