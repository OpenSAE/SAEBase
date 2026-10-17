"""Real-checkpoint external-extension probe for Qwen3-1.7B Base L7.

A passing probe checks one actual checkpoint and hook, not population
reliability or general checkpoint certification. Float32 matches the
declared external SAE contract; Stage D uses a separate bfloat16 profile.
"""

from __future__ import annotations

import json

import torch
from transformers import AutoTokenizer

from models.hf_hooks_backend import load_hf_hooks_model
from saebase.api import ActivationSite, capture_features, edit_features
from saebase.extensions import (
    ExtensionBundle,
    ExternalModel,
    ExternalSAE,
    HookSpec,
    load_external_pair,
    register_extension,
)
from saebase.sae.loading import load_sae_for_profile

MODEL_REPO = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SAE_REPO = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
SAE_REVISION = "ce1a79d9c5163932d65c417380e53230e1086370"
LAYER = 7
SITE = ActivationSite(LAYER, "blocks.7.hook_resid_post", position=-1)
PROMPT = "The capital of France is"


def load_model(device: str):
    handle = load_hf_hooks_model(
        MODEL_REPO,
        key="qwen3_1_7b_base",
        family="qwen",
        device=device,
        dtype=torch.float32,
        revision=MODEL_REVISION,
    )
    return handle.model


def load_sae(device: str):
    sae, _meta = load_sae_for_profile(
        "qwen3_1_7b_base_resid_w32k",
        LAYER,
        device=device,
        revision=SAE_REVISION,
    )
    return sae


def main() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    register_extension(
        ExtensionBundle(
            name="ext.opensae.qwen3_1_7b_l7",
            models=(
                ExternalModel(
                    key="ext.opensae.qwen3_1_7b_base",
                    source=f"https://huggingface.co/{MODEL_REPO}",
                    revision=MODEL_REVISION,
                    hooks=(HookSpec(SITE, 2048),),
                    factory=load_model,
                    backend="hf_hooks",
                ),
            ),
            saes=(
                ExternalSAE(
                    key="ext.opensae.qwen3_1_7b_base_l7",
                    model_key="ext.opensae.qwen3_1_7b_base",
                    source=f"https://huggingface.co/{SAE_REPO}",
                    revision=SAE_REVISION,
                    site=SITE,
                    d_in=2048,
                    d_sae=32768,
                    architecture="topk",
                    normalization="none",
                    dtype="float32",
                    sae_id="layer7.sae.pt",
                    factory=load_sae,
                ),
            ),
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    tokens = tokenizer(PROMPT, return_tensors="pt").input_ids
    if device == "cuda":
        tokens = tokens.cuda()

    pair = load_external_pair(
        model_key="ext.opensae.qwen3_1_7b_base",
        sae_key="ext.opensae.qwen3_1_7b_base_l7",
        layer=LAYER,
        device=device,
        probe_tokens=tokens,
    )
    snap = capture_features(pair.model, pair.sae, site=SITE, tokens=tokens)
    selected = int(torch.argmax(snap.codes.abs()).item())
    edited = edit_features(pair.sae, snap, replacements={selected: 0.0})
    delta = float(
        torch.linalg.vector_norm(
            edited.activation.float() - snap.activation.float()
        ).item()
    )
    record = {
        "schema": "saebase.stage_d_recovery_03.external_interop.v1",
        "status": pair.verification,
        "claim": "adapter interoperability on this exact checkpoint and site only",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "sae_repo": SAE_REPO,
        "sae_revision": SAE_REVISION,
        "layer": LAYER,
        "hook_name": SITE.hook_name,
        "d_in": 2048,
        "d_sae": 32768,
        "dtype": "float32",
        "normalization": "none",
        "l0": int(torch.count_nonzero(snap.codes).item()),
        "ablated_feature_id": selected,
        "activation_delta_l2": delta,
        "finite": bool(
            torch.isfinite(snap.reconstruction).all()
            and torch.isfinite(edited.activation).all()
        ),
        "nontrivial_ablation": delta > 1e-8,
    }
    if (
        pair.verification != "probe_passed_not_checkpoint_certified"
        or not record["nontrivial_ablation"]
    ):
        raise SystemExit(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
