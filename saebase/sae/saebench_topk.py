"""Pinned original SAEBench Gemma-2-2B L12 TopK dictionary_learning checkpoint.

This adapter uses SAEBench's published TopKSAE class rather than implementing
or changing SAE mathematics. The original Torch weights are decoded with
weights_only=True and validated before constructing the upstream SAE.
Network access is opt-in through load_sae_for_profile, never at import time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO = "canrager/saebench_gemma-2-2b_width-2pow14_date-0107"
REVISION = "18190d53776b0710543f8e7e8b78a2cb8f12e773"
FOLDER = "gemma-2-2b_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2"
MODEL = "google/gemma-2-2b"
HOOK = "blocks.12.hook_resid_post"
D_IN = 2304
D_SAE = 16384
K = 80


def validate_source(
    repo_id: str,
    checkpoint_dir: str,
    expected_layer: int,
    expected_hook: str,
    revision: str | None,
) -> None:
    """Refuse a new or mismatched asset rather than guessing compatibility."""
    if (
        repo_id != REPO
        or checkpoint_dir != FOLDER
        or expected_layer != 12
        or expected_hook != HOOK
        or revision not in (None, REVISION)
    ):
        raise ValueError("SAEBench baseline source identity is not approved")
    if not re.fullmatch(r"[0-9a-f]{40}", REVISION):
        raise ValueError("SAEBench baseline must pin a full source commit")


def validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("trainer"), dict):
        raise ValueError("SAEBench config missing trainer metadata")
    trainer = value["trainer"]
    expected = {
        "trainer_class": "TopKTrainer",
        "dict_class": "AutoEncoderTopK",
        "lm_name": MODEL,
        "layer": 12,
        "submodule_name": "resid_post_layer_12",
        "activation_dim": D_IN,
        "dict_size": D_SAE,
        "k": K,
    }
    if any(trainer.get(k) != v for k, v in expected.items()):
        raise ValueError("SAEBench model/architecture/layer/width/k mismatch")
    return trainer


def remap_state_dict(raw: Any) -> dict[str, Any]:
    """Map the dictionary_learning tensor layout exactly as upstream does."""
    import torch

    if not isinstance(raw, dict):
        raise TypeError("SAEBench checkpoint must contain a tensor state dict")
    needed = {"encoder.weight", "decoder.weight", "encoder.bias", "bias", "k"}
    if not needed.issubset(raw) or set(raw) - needed - {"threshold"}:
        raise ValueError("SAEBench TopK checkpoint tensor keys mismatch")
    if not all(isinstance(value, torch.Tensor) for value in raw.values()):
        raise TypeError("SAEBench checkpoint contains non-tensor entries")
    if (
        tuple(raw["encoder.weight"].shape) != (D_SAE, D_IN)
        or tuple(raw["decoder.weight"].shape) != (D_IN, D_SAE)
        or tuple(raw["encoder.bias"].shape) != (D_SAE,)
        or tuple(raw["bias"].shape) != (D_IN,)
        or raw["k"].numel() != 1
        or int(raw["k"].item()) != K
    ):
        raise ValueError("SAEBench TopK checkpoint tensor shape/k mismatch")
    if any(not torch.isfinite(value).all().item() for name, value in raw.items() if name != "k"):
        raise ValueError("SAEBench checkpoint has non-finite weights")
    # Source TopK inference intentionally ignores the training threshold.
    # Keep the original upstream key mapping and transpositions.
    return {
        "W_enc": raw["encoder.weight"].T.contiguous(),
        "W_dec": raw["decoder.weight"].T.contiguous(),
        "b_enc": raw["encoder.bias"],
        "b_dec": raw["bias"],
        "k": raw["k"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_saebench_topk(
    *,
    repo_id: str,
    checkpoint_dir: str,
    expected_layer: int,
    expected_hook: str,
    device: str,
    revision: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Explicitly download/load one immutable native SAE Bench baseline."""
    validate_source(repo_id, checkpoint_dir, expected_layer, expected_hook, revision)

    import torch
    from huggingface_hub import hf_hub_download

    # Only the original published TopKSAE implementation is used for inference.
    try:
        from sae_bench.custom_saes.topk_sae import TopKSAE
    except ImportError as exc:
        raise RuntimeError(
            "Install the approved upstream SAEBench package for this optional baseline"
        ) from exc

    cfg_file = hf_hub_download(repo_id=REPO, filename=f"{FOLDER}/config.json", revision=REVISION)
    config = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
    validate_config(config)
    weights_file = hf_hub_download(repo_id=REPO, filename=f"{FOLDER}/ae.pt", revision=REVISION)
    raw = torch.load(weights_file, map_location="cpu", weights_only=True)
    state = remap_state_dict(raw)

    sae = TopKSAE(
        d_in=D_IN,
        d_sae=D_SAE,
        k=K,
        model_name="gemma-2-2b",
        hook_layer=12,
        device=torch.device(device),
        dtype=torch.bfloat16,
        use_threshold=False,
        hook_name=HOOK,
    )
    sae.load_state_dict(state, strict=True)
    sae.to(device=torch.device(device), dtype=torch.bfloat16)
    sae.cfg.architecture = "topk"
    if sae.cfg.hook_name != HOOK or not sae.check_decoder_norms():
        raise ValueError("Native SAEBench SAE hook/decoder normalization mismatch")
    sae.eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae, {
        "config": config,
        "revision": REVISION,
        "checkpoint_sha256": _sha256(Path(weights_file)),
    }
