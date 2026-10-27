"""Activation-health evaluation: last-token hook + SAE encode/decode."""

from __future__ import annotations

import gc
from typing import Any

from omegaconf import DictConfig, OmegaConf

from models.hooked_lm_common import print_vram_gb
from tasks.benchmark.health_results import EvaluationResult
from tasks.benchmark.analyze import analyze_prompt_activation
from saebase.sae.loading import load_model_and_sae
from tasks.benchmark.pipeline_checks import run_preflight
from research_datasets.legacy_prompts import load_behavior_prompts
from tasks.benchmark.health_results import summarize_rows, write_run_summary


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
        "task": "sae",
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
        task="sae",
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
