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
    converter: Any | None = None,
) -> tuple[Any, dict, Any]:
    SAE = import_sae_lens().SAE
    dev = device or require_cuda()
    legacy = getattr(SAE, "from_pretrained_with_cfg_and_sparsity", None)
    loader = legacy if callable(legacy) else SAE.from_pretrained
    kwargs = {"release": release, "sae_id": sae_id, "device": dev}
    if converter is not None:
        kwargs["converter"] = converter
    out = loader(**kwargs)
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
    """Build a TopKSAE from an official Qwen residual ``layer{L}.sae.pt`` blob."""
    sl = import_sae_lens()
    TopKSAE = sl.saes.TopKSAE
    TopKSAEConfig = sl.saes.TopKSAEConfig
    SAEMetadata = sl.saes.sae.SAEMetadata

    # Hugging Face is only needed for remote checkpoint loading.
    from huggingface_hub import hf_hub_download

    dev = device or require_cuda()
    fname = filename or f"layer{int(layer)}.sae.pt"
    weights_path = hf_hub_download(repo_id=release, filename=fname)
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"{release}/{fname}: expected a state dict, got {type(blob)}")

    def _first(*names: str) -> torch.Tensor:
        for name in names:
            if name in blob:
                return blob[name]
        raise KeyError(f"{release}/{fname}: missing keys among {names}")

    w_enc = _first("W_enc", "w_enc", "encoder.weight")
    b_enc = _first("b_enc", "encoder.bias")
    w_dec = _first("W_dec", "w_dec", "decoder.weight")
    b_dec = blob.get("b_dec", blob.get("decoder.bias"))
    if b_dec is None:
        b_dec = torch.zeros(w_dec.shape[-1] if w_dec.ndim == 2 else int(w_enc.shape[-1]))

    if w_enc.ndim != 2 or w_dec.ndim != 2:
        raise ValueError(f"{release}/{fname}: W_enc/W_dec must be rank-2")
    if w_enc.shape[0] > w_enc.shape[1]:
        # Official Qwen dumps are (d_sae, d_in).
        d_sae, d_in = int(w_enc.shape[0]), int(w_enc.shape[1])
        w_enc_tl = w_enc.T.contiguous()
    else:
        d_in, d_sae = int(w_enc.shape[0]), int(w_enc.shape[1])
        w_enc_tl = w_enc.contiguous()
    if tuple(w_dec.shape) == (d_sae, d_in):
        w_dec_tl = w_dec
    elif tuple(w_dec.shape) == (d_in, d_sae):
        w_dec_tl = w_dec.T.contiguous()
    else:
        raise ValueError(
            f"{release}/{fname}: W_dec shape {tuple(w_dec.shape)} "
            f"incompatible with d_in={d_in} d_sae={d_sae}"
        )

    k_val = int(blob.get("k", blob.get("top_k", k)))
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
        "W_enc": w_enc_tl.float(),
        "b_enc": b_enc.float().reshape(-1),
        "W_dec": w_dec_tl.float(),
        "b_dec": b_dec.float().reshape(-1),
    }
    sae.load_state_dict(state, strict=True)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    raw_cfg = {
        "d_in": d_in,
        "d_sae": d_sae,
        "k": k_val,
        "filename": fname,
        "repo_id": release,
    }
    return sae, raw_cfg, None


def resolve_hook_name(sae: Any, profile: SAEProfile, layer: int) -> str:
    hook = getattr(sae.cfg, "hook_name", None)
    if hook:
        return str(hook)
    return profile.hook_name(layer)


def _phase_a_sae_lens_source(profile: SAEProfile, layer: int) -> tuple[str, str, str | None]:
    """Resolve exact public SAE source paths independently of SAELens release aliases.

    The official Gemma Scope 2 SAEs use config.json + params.safetensors,
    and Llama Scope uses hyperparams.json + checkpoints/final.safetensors.
    SAELens' named release directory may lag the public repositories.
    """
    if profile.key in {"gemma270m_scope_resid", "gemma1b_scope_resid",
                       "gemma3_4b_pt_scope_resid", "gemma3_4b_it_scope_resid"}:
        return (
            str(profile.repo_id),
            f"resid_post/{profile.sae_id(layer)}",
            "gemma_3_sae_huggingface_loader",
        )
    if profile.key in {"r1_distill_llama8b_llamascope_l23",
                       "r1_distill_llama8b_llamascope_l15_openr1"}:
        return (str(profile.repo_id), str(profile.sae_folder(layer)),
                "llama_scope_r1_distill_sae_huggingface_loader")
    if profile.key == "mistral7b_instruct_tc_l16_131k":
        return (str(profile.repo_id), ".", "sae_lens_huggingface_loader")
    if profile.key == "llama31_8b_lxr_8x":
        return (
            str(profile.repo_id),
            profile.sae_id(layer),
            "llama_scope_sae_huggingface_loader",
        )
    return profile.sae_release, profile.sae_id(layer), None


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
        source_release, source_id, converter_name = _phase_a_sae_lens_source(profile, layer)
        converter = None
        if converter_name is not None:
            from sae_lens.loading import pretrained_sae_loaders
            converter = getattr(pretrained_sae_loaders, converter_name)
        sae, cfg_dict, sparsity = load_sae_lens(
            release=source_release,
            sae_id=source_id,
            device=dev,
            converter=converter,
        )
        # Direct Llama-Scope repository IDs are not in older SAE Lens release
        # directories, so its upstream norm-scaling lookup returns None.
        # Recover the *published* expected activation norm from hyperparams.
        # The native loader already scales the JumpReLU threshold by the same
        # factor; fold only W_enc/W_dec/b_dec to match raw HF/TransformerLens h.
        if profile.key == "llama31_8b_lxr_8x" and cfg_dict.get(
            "normalize_activations"
        ) == "expected_average_only_in":
            import math
            from huggingface_hub import hf_hub_download
            params = hf_hub_download(
                repo_id=source_release,
                filename=f"{source_id}/hyperparams.json",
            )
            original = json.loads(Path(params).read_text(encoding="utf-8"))
            mean_norm = float(original["dataset_average_activation_norm"]["in"])
            scaling = math.sqrt(float(sae.cfg.d_in)) / mean_norm
            if not math.isfinite(scaling) or scaling <= 0:
                raise ValueError("invalid published Llama Scope input activation norm")
            sae.fold_activation_norm_scaling_factor(scaling)
            cfg_dict["normalize_activations"] = "none"
            cfg_dict["saebase_activation_norm_folded"] = scaling
        hook_name = resolve_hook_name(sae, profile, layer)
        meta: dict[str, Any] = {
            "sae_key": profile.key,
            "model_key": profile.model_key,
            "model_name": profile.model_name,
            "layer": layer,
            "sae_release": profile.sae_release,
            "sae_id": sae_id,
            "resolved_source_release": source_release,
            "resolved_source_sae_id": source_id,
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
        if profile.key.startswith("llama31_8b_instruct_andyrdt"):
            raise RuntimeError(
                "The original Llama3.1 instruct BatchTopK checkpoint still requires "
                "the SAEReasoning protocol-10 loader; no independent real weight "
                "acceptance has been performed through SAEBase."
            )
        from sae_lens.loading import pretrained_sae_loaders
        converter = pretrained_sae_loaders.dictionary_learning_sae_huggingface_loader_1
        sae, cfg_dict, sparsity = load_sae_lens(
            release=str(profile.repo_id or profile.sae_release),
            sae_id=str(folder or sae_id), device=dev, converter=converter,
        )
        actual_hook = resolve_hook_name(sae, profile, layer)
        if actual_hook != hook_name:
            raise ValueError("dictionary_learning SAE source hook does not match layer")
        return sae, {
            "sae_key": profile.key, "model_key": profile.model_key,
            "model_name": profile.model_name, "layer": layer,
            "sae_release": profile.sae_release, "sae_id": sae_id,
            "hook_name": actual_hook, "loader": profile.loader,
            "folder": folder, "cfg": cfg_dict, "sparsity": sparsity,
            "d_in": int(sae.cfg.d_in), "d_sae": int(sae.cfg.d_sae),
            **profile.transfer_meta(),
        }

    if profile.loader == "sparsify_topk":
        from .sparsify_adapter import load_sparsify_topk
        sae, cfg_dict = load_sparsify_topk(
            repo_id=str(profile.repo_id or profile.sae_release),
            hookpoint=str(folder or sae_id), hook_name=hook_name, device=dev,
        )
        return sae, {
            "sae_key": profile.key, "model_key": profile.model_key,
            "model_name": profile.model_name, "layer": layer,
            "sae_release": profile.sae_release, "sae_id": sae_id,
            "hook_name": hook_name, "loader": profile.loader,
            "folder": folder, "cfg": cfg_dict, "sparsity": None,
            "d_in": int(sae.cfg.d_in), "d_sae": int(sae.cfg.d_sae),
            **profile.transfer_meta(),
        }

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
