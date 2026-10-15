"""Unified SAE loaders keyed by ``saebase.sae.registry`` profiles."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch

from models.hooked_lm_common import (
    load_hooked_model,
    load_hooked_model_with_candidates,
    require_cuda,
)
from models.hf_compat import import_sae_lens
from saebase.sae.registry import SAEProfile, get_profile


def get_hook_name(sae_key: str, layer: int) -> str:
    return get_profile(sae_key).hook_name(layer)


def get_sae_id(sae_key: str, layer: int) -> str:
    return get_profile(sae_key).sae_id(layer)


def load_sae_lens(
    *,
    release: str,
    sae_id: str,
    device: str | None = None,
) -> tuple[Any, dict, Any]:
    SAE = import_sae_lens().SAE
    dev = device or require_cuda()
    legacy = getattr(SAE, "from_pretrained_with_cfg_and_sparsity", None)
    loader = legacy if callable(legacy) else SAE.from_pretrained
    out = loader(release=release, sae_id=sae_id, device=dev)
    if isinstance(out, tuple):
        if not out:
            raise RuntimeError(f"SAE loader returned an empty tuple for {release}/{sae_id}")
        sae = out[0]
        cfg_dict = out[1] if len(out) > 1 else {}
        sparsity = out[2] if len(out) > 2 else None
    else:
        sae = out
        cfg_dict, sparsity = {}, None
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    return sae, cfg_dict, sparsity


def load_eleutherai_topk_mlp_sae(
    *,
    release: str,
    layer: int,
    folder: str,
    hook_name: str,
    model_name: str,
    device: str | None = None,
    k: int = 32,
) -> tuple[Any, dict, None]:
    """Build a TopKSAE from EleutherAI per-layer MLP checkpoint (Pythia-410M)."""
    sl = import_sae_lens()
    TopKSAE = sl.saes.TopKSAE
    TopKSAEConfig = sl.saes.TopKSAEConfig
    SAEMetadata = sl.saes.sae.SAEMetadata

    # These packages belong to the optional analysis extra, not the base API.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    dev = device or require_cuda()
    cfg_path = hf_hub_download(repo_id=release, filename=f"{folder}/cfg.json")
    weights_path = hf_hub_download(
        repo_id=release, filename=f"{folder}/sae.safetensors"
    )
    raw_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    st = load_file(weights_path)

    d_in = int(raw_cfg["d_in"])
    d_sae = int(raw_cfg["num_latents"])
    k_val = int(raw_cfg.get("k", k))

    cfg = TopKSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k_val,
        device=dev,
        dtype="float32",
        metadata=SAEMetadata(
            model_name=model_name,
            hook_name=hook_name,
        ),
    )
    sae = TopKSAE(cfg)
    state = {
        "W_enc": st["encoder.weight"].T.contiguous(),
        "b_enc": st["encoder.bias"],
        "W_dec": st["W_dec"],
        "b_dec": st["b_dec"],
    }
    sae.load_state_dict(state, strict=True)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, raw_cfg, None


def load_qwen_resid_topk_sae(
    *,
    release: str,
    layer: int,
    hook_name: str,
    model_name: str,
    device: str | None = None,
    k: int = 50,
    filename: str | None = None,
) -> tuple[Any, dict, None]:
    """Load official Qwen-Scope TopK tensors without requiring SAE Lens."""
    from huggingface_hub import hf_hub_download
    from saebase.sae.qwen_topk import from_official_checkpoint

    dev = device or require_cuda()
    fname = filename or f"layer{int(layer)}.sae.pt"
    if fname != f"layer{int(layer)}.sae.pt":
        raise ValueError("Qwen-Scope SAE ID must identify the requested layer file")
    weights_path = hf_hub_download(repo_id=release, filename=fname)
    expected = {
        "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50": (2048, 32768),
        "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50": (4096, 65536),
    }.get(release)
    sae = from_official_checkpoint(
        weights_path, hook_name=hook_name, device=dev,
        expected_hidden=expected[0] if expected else None,
        expected_features=expected[1] if expected else None,
        k=k,
    )
    return sae, {
        "d_in": sae.cfg.d_in,
        "d_sae": sae.cfg.d_sae,
        "k": sae.cfg.k,
        "filename": fname,
        "repo_id": release,
        "model_name": model_name,
        "format": "official-qwen-scope-topk",
    }, None


def resolve_hook_name(sae: Any, profile: SAEProfile, layer: int) -> str:
    hook = getattr(sae.cfg, "hook_name", None)
    if hook:
        return str(hook)
    return profile.hook_name(layer)


def load_sae_for_profile(
    sae_key: str,
    layer: int,
    device: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    profile = get_profile(sae_key)
    dev = device or require_cuda()
    hook_name = profile.hook_name(layer)
    sae_id = profile.sae_id(layer)
    folder = profile.sae_folder(layer)

    if profile.loader == "sae_lens":
        sae, cfg_dict, sparsity = load_sae_lens(
            release=profile.sae_release,
            sae_id=sae_id,
            device=dev,
        )
        hook_name = resolve_hook_name(sae, profile, layer)
        meta: dict[str, Any] = {
            "sae_key": profile.key,
            "model_key": profile.model_key,
            "model_name": profile.model_name,
            "layer": layer,
            "sae_release": profile.sae_release,
            "sae_id": sae_id,
            "hook_name": hook_name,
            "hook_note": (
                profile.hook_note.replace("{L}", str(layer))
                if profile.hook_note
                else f"{hook_name} (pre-final-norm)"
                if profile.hook_kind == "hook_resid_post"
                else hook_name
            ),
            "loader": profile.loader,
            "folder": folder,
            "cfg": cfg_dict,
            "sparsity": sparsity,
            "d_in": int(sae.cfg.d_in),
            "d_sae": int(sae.cfg.d_sae),
            **profile.transfer_meta(),
        }
        return sae, meta

    if profile.loader == "eleutherai_topk_mlp":
        if folder is None:
            raise RuntimeError(f"{profile.key}: missing sae_folder_template")
        sae, raw_cfg, _ = load_eleutherai_topk_mlp_sae(
            release=profile.sae_release,
            layer=layer,
            folder=folder,
            hook_name=hook_name,
            model_name=profile.model_name,
            device=dev,
        )
        meta = {
            "sae_key": profile.key,
            "model_key": profile.model_key,
            "model_name": profile.model_name,
            "layer": layer,
            "sae_release": profile.sae_release,
            "sae_id": sae_id,
            "hook_name": hook_name,
            "loader": profile.loader,
            "folder": folder,
            "cfg": raw_cfg,
            "sparsity": None,
            "d_in": int(sae.cfg.d_in),
            "d_sae": int(sae.cfg.d_sae),
        }
        return sae, meta

    if profile.loader == "qwen_resid_topk":
        sae, raw_cfg, _ = load_qwen_resid_topk_sae(
            release=profile.sae_release,
            layer=layer,
            hook_name=hook_name,
            model_name=profile.model_name,
            device=dev,
            filename=sae_id,
        )
        meta = {
            "sae_key": profile.key,
            "model_key": profile.model_key,
            "model_name": profile.model_name,
            "layer": layer,
            "sae_release": profile.sae_release,
            "sae_id": sae_id,
            "hook_name": hook_name,
            "loader": profile.loader,
            "folder": folder,
            "cfg": raw_cfg,
            "sparsity": None,
            "d_in": int(sae.cfg.d_in),
            "d_sae": int(sae.cfg.d_sae),
        }
        return sae, meta

    if profile.loader == "dictionary_learning":
        raise RuntimeError(
            "The dictionary_learning BatchTopK checkpoint needs the "
            "external SAEReasoning protocol-10 loader, which is not included "
            "with SAEBase Phase A. Use a verified sae_lens checkpoint for "
            "the standalone activation-health workflow."
        )

    raise ValueError(f"Unknown loader kind: {profile.loader!r}")


def load_model_for_profile(
    sae_key: str,
    device: str | None = None,
    *,
    model_key_override: str | None = None,
) -> tuple[Any, str]:
    """Load the hooked LM for an SAE profile.

    ``model_key_override`` supports transfer diagnostics (PT SAE on IT host).
    """
    profile = get_profile(sae_key)
    mk = model_key_override or profile.model_key
    try:
        mod = importlib.import_module(f"models.{mk}.{mk}")
        handle = mod.load_model(device=device)
        return handle.model, handle.loaded_name
    except ModuleNotFoundError as exc:
        # Only fall back when the requested model wrapper itself is absent.
        # A missing dependency inside a present wrapper should retain its name.
        if exc.name not in {f"models.{mk}", f"models.{mk}.{mk}"}:
            raise

    candidates = profile.model_candidates or (profile.model_name,)
    if len(candidates) == 1:
        model = load_hooked_model(candidates[0], device=device)
        return model, candidates[0]
    return load_hooked_model_with_candidates(candidates, device=device)


def load_model_and_sae(
    sae_key: str,
    layer: int,
    device: str | None = None,
    *,
    model_key_override: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    dev = device or require_cuda()
    model, loaded_name = load_model_for_profile(
        sae_key, device=dev, model_key_override=model_key_override
    )
    sae, meta = load_sae_for_profile(sae_key, layer, device=dev)
    meta = {**meta, "loaded_model_name": loaded_name}
    return model, sae, meta


def sae_encode_decode_smoke(
    sae: Any, device: str, batch: int = 4
) -> dict[str, float | int]:
    x = torch.randn(batch, sae.cfg.d_in, device=device)
    with torch.no_grad():
        z = sae.encode(x)
        xhat = sae.decode(z)
    mse = float(((x - xhat) ** 2).mean().item())
    return {
        "d_in": int(sae.cfg.d_in),
        "d_sae": int(sae.cfg.d_sae),
        "z_shape_1": int(z.shape[1]),
        "recon_mse_random": mse,
    }


def activation_encode_smoke(
    model: Any,
    sae: Any,
    hook_name: str,
    text: str = "The capital of France is",
) -> dict[str, float | int]:
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        h = cache[hook_name][0, -1]
        z = sae.encode(h.unsqueeze(0))[0]
    nnz = int((z != 0).sum().item())
    return {
        "d_in": int(sae.cfg.d_in),
        "d_sae": int(sae.cfg.d_sae),
        "activation_norm": float(h.norm().item()),
        "l0": nnz,
    }
