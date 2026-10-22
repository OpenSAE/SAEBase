"""Load authentic Gemma Scope 2 JumpReLU SAEs, transcoders and crosscoders.

Uses the official config.json and safetensors at an immutable Hugging Face
revision. Does not guess the type, skip connection or hook coordinates, and
does not load unsafe pickle. Large checkpoints are opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import nn


_REPO = re.compile(r"^google/gemma-scope-2-(270m|1b|4b|12b|27b)-(pt|it)$")
_SHA = re.compile(r"^[a-f0-9]{40}$")
_DIR = re.compile(r"^[a-z0-9_]+/[a-zA-Z0-9_]+$")
_ALLOWED = {
    "resid_post", "resid_post_all", "attn_out", "attn_out_all",
    "mlp_out", "mlp_out_all", "transcoder", "transcoder_all",
    "crosscoder", "clt",
}


@dataclass(frozen=True)
class ScopeIdentity:
    repo_id: str
    revision: str
    asset_directory: str
    family: str
    kind: str
    input_hook: str
    output_hook: str
    source_sha256: str
    width: int
    input_dim: int
    output_dim: int
    affine_skip: bool
    source_model: str


class JumpReLURepresentation(nn.Module):
    """Exact basic Gemma Scope 2 parameterization; no processed-weight switch.

    Multi-input partial crosscoders are accepted when w_enc is [L,D,F] and
    w_dec is [F,L,D] or [L,F,D]. Unknown CLT tensor contractions are NOT
    silently interpreted as ordinary SAEs.
    """

    def __init__(
        self, params: Mapping[str, torch.Tensor], *,
        config: Mapping[str, Any],
    ):
        super().__init__()
        required = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        if not required.issubset(params):
            raise ValueError("Official JumpReLU parameter set incomplete")
        if config.get("architecture") != "jump_relu":
            raise ValueError("Only official JumpReLU architecture supported")
        w_enc, w_dec = params["w_enc"], params["w_dec"]
        if not all(isinstance(params[k], torch.Tensor) for k in required):
            raise TypeError("Safe tensor weights required")
        if w_enc.ndim not in (2, 3):
            raise ValueError("Unsupported native encoder rank")
        features = int(w_enc.shape[-1])
        input_dim = int(w_enc.shape[-2])
        if (
            not 1 <= features <= 1_200_000
            or not 1 <= input_dim <= 16384
            or int(config.get("width", -1)) != features
            or tuple(params["b_enc"].shape) != (features,)
            or tuple(params["threshold"].shape) != (features,)
        ):
            raise ValueError("Native feature dimensions/config do not agree")
        if w_enc.ndim == 2:
            if tuple(w_dec.shape) != (features, input_dim):
                raise ValueError("Single-layer decoder does not match native encoder")
            if tuple(params["b_dec"].shape) != (input_dim,):
                raise ValueError("Single-layer decoder bias mismatch")
        else:
            layer_count = int(w_enc.shape[0])
            if tuple(w_dec.shape) not in {
                (features, layer_count, input_dim),
                (layer_count, features, input_dim),
            }:
                raise ValueError("Multi-layer decoder orientation is unsupported")
            if tuple(params["b_dec"].shape) != (layer_count, input_dim):
                raise ValueError("Multi-layer output bias does not match native sites")
        affine = bool(config.get("affine_connection", False))
        if affine and w_enc.ndim != 2:
            raise ValueError("Multi-layer affine skip requires a separate native contract")
        if affine:
            if "affine_skip_connection" not in params or tuple(
                params["affine_skip_connection"].shape
            ) != (input_dim, input_dim):
                raise ValueError("Affine transcoder requires exact skip matrix")
        elif "affine_skip_connection" in params:
            raise ValueError("Unexpected affine matrix disagrees with original config")
        permitted = required | {"affine_skip_connection"}
        if set(params) != permitted if affine else set(params) != required:
            raise ValueError("Unknown source tensors; refuse undocumented normalization")
        if any(not bool(torch.isfinite(weight).all()) for weight in params.values()):
            raise ValueError("Nonfinite native parameters")
        for key, weight in params.items():
            self.register_buffer(key, weight.detach().clone())
        self.input_dim = input_dim
        self.num_features = features
        self.multi_layer = w_enc.ndim == 3
        self.affine = affine

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_layer:
            if x.shape[-2:] != self.w_enc.shape[:2]:
                raise ValueError("Crosscoder input must align all declared input layers")
            pre = torch.einsum("...ld,ldf->...f", x, self.w_enc) + self.b_enc
        else:
            if x.shape[-1] != self.input_dim:
                raise ValueError("Wrong native input activation dimension")
            pre = x @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, torch.relu(pre), torch.zeros_like(pre))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.num_features:
            raise ValueError("Wrong JumpReLU feature-code width")
        if self.multi_layer:
            if self.w_dec.shape[0] == self.num_features:
                return torch.einsum("...f,fld->...ld", z, self.w_dec) + self.b_dec
            return torch.einsum("...f,lfd->...ld", z, self.w_dec) + self.b_dec
        return z @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted = self.decode(self.encode(x))
        if self.affine:
            predicted = predicted + x @ self.affine_skip_connection
        return predicted


def _validate_source(repo_id: str, revision: str, directory: str) -> str:
    if not _REPO.fullmatch(repo_id) or not _SHA.fullmatch(revision):
        raise ValueError("Use an official Gemma Scope 2 repository and immutable revision")
    if not _DIR.fullmatch(directory):
        raise ValueError("Use an exact official asset folder, no arbitrary paths")
    family = directory.split("/")[0]
    if family not in _ALLOWED:
        raise ValueError("Unrecognized official representation family")
    return family


def load_official_scope2(
    *, repo_id: str, revision: str, directory: str, device: str = "cpu",
    max_checkpoint_bytes: int = 1_000_000_000,
) -> tuple[JumpReLURepresentation, ScopeIdentity]:
    """Download two named official files and load safetensors only, with fail-closed identity.

    Large width variants need a separate memory budget. No extra files are
    downloaded and the revision never floats to main.
    """
    family = _validate_source(repo_id, revision, directory)
    if not 1 <= max_checkpoint_bytes <= 12_000_000_000:
        raise ValueError("Explicit positive weight budget required")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = Path(hf_hub_download(
        repo_id=repo_id, filename=f"{directory}/config.json", revision=revision
    ))
    weights_path = Path(hf_hub_download(
        repo_id=repo_id, filename=f"{directory}/params.safetensors", revision=revision
    ))
    if config_path.stat().st_size > 16_384 or weights_path.stat().st_size > max_checkpoint_bytes:
        raise ValueError("Official checkpoint exceeds declared memory/storage budget")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Invalid official asset config")
    family_kind = {
        "crosscoder": "crosscoder",
        "clt": "crosslayer",
        "transcoder": "transcoder",
        "transcoder_all": "transcoder",
    }
    kind = family_kind.get(family, "sae")
    actual_kind = config.get("type", kind)
    if actual_kind not in {kind, None}:
        raise ValueError("Published source type differs from the asset family")
    incoming, outgoing = config.get("hf_hook_point_in"), config.get("hf_hook_point_out")
    if not all(isinstance(s, str) and s for s in (incoming, outgoing)):
        raise ValueError("Official checkpoint does not declare both hook locations")
    if kind == "sae" and incoming != outgoing:
        raise ValueError("A same-site SAE may not reconstruct another hook")
    if kind == "transcoder" and incoming == outgoing:
        raise ValueError("A transcoder must map a different input to output hook")
    params = load_file(str(weights_path), device="cpu")
    net = JumpReLURepresentation(params, config=config).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if device != "cpu":
        net = net.to(device)
    source_hash = hashlib.sha256()
    for path in (config_path, weights_path):
        with path.open("rb") as stream:
            for buf in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                source_hash.update(buf)
    output_dim = int(net.b_dec.shape[-1])
    info = ScopeIdentity(
        repo_id=repo_id, revision=revision, asset_directory=directory,
        family=family, kind=kind, input_hook=incoming, output_hook=outgoing,
        source_sha256=source_hash.hexdigest(), width=net.num_features,
        input_dim=net.input_dim, output_dim=output_dim, affine_skip=net.affine,
        source_model=str(config.get("model_name", "")),
    )
    return net, info
