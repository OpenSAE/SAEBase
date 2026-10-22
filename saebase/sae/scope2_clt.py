"""Native Gemma Scope 2 cross-layer transcoder (CLT) shard algebra.

A feature at source layer i decodes to *each target layer j >= i*. Official
files are params_layer_i.safetensors and contain per-source w_enc [D,F],
w_dec [F,L,D], biases and JumpReLU threshold. No flattened SAE shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .scope2_native import _validate_source


@dataclass(frozen=True)
class CLTIdentity:
    repo_id: str
    revision: str
    asset_directory: str
    input_hook_template: str
    output_hook_template: str
    model_name: str
    source_sha256: str
    n_layers: int
    d_model: int
    features_per_layer: int
    affine_skip: bool


class CLTLayer(nn.Module):
    """Per-source original weights; retain target dimension of w_dec."""

    def __init__(
        self, params: Mapping[str, torch.Tensor], *,
        n_layers: int, n_features: int, d_model: int, affine: bool,
    ):
        super().__init__()
        expected = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        if affine:
            expected.add("affine_skip_connection")
        if set(params) != expected:
            raise ValueError("Native CLT shard tensor keys do not match config")
        shape = {
            "w_enc": (d_model, n_features),
            "w_dec": (n_features, n_layers, d_model),
            "b_enc": (n_features,),
            "b_dec": (d_model,),
            "threshold": (n_features,),
        }
        if affine:
            shape["affine_skip_connection"] = (d_model, d_model)
        for key, dims in shape.items():
            tensor = params[key]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != dims:
                raise ValueError(f"CLT source tensor {key} disagrees with official layer shape")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Native CLT weights must be finite")
            self.register_buffer(key, tensor.detach().clone())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.w_enc.shape[0]:
            raise ValueError("CLT input residual dimension mismatch")
        x = x.to(device=self.w_enc.device, dtype=self.w_enc.dtype)
        pre = x @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, torch.relu(pre), torch.zeros_like(pre))

    def output_delta(
        self, x: torch.Tensor, *, source_layer: int,
        feature_id: int, new_code: float,
    ) -> torch.Tensor:
        """Only this selected feature changes; no bias/skip/other codes."""
        if not 0 <= feature_id < self.b_enc.shape[0]:
            raise IndexError("Selected CLT feature index unavailable")
        code = self.encode(x)
        old = code[..., feature_id]
        step = (torch.as_tensor(new_code, device=old.device, dtype=old.dtype) - old)
        # [batch, n_target, d_model], with no effects before the source layer.
        delta = step[:, None, None] * self.w_dec[feature_id][None, :, :]
        delta[:, :source_layer, :] = 0
        return delta


class Scope2CLT(nn.Module):
    """Real native cross-layer JumpReLU encoder/decoder with triangular writes."""

    def __init__(
        self, layers: Sequence[Mapping[str, torch.Tensor]],
        *, config: Mapping[str, Any],
    ):
        super().__init__()
        count = len(layers)
        if not 2 <= count <= 64 or config.get("architecture") != "jump_relu":
            raise ValueError("Native CLT layer count and architecture required")
        required_hook = "model.layers.{all}."
        if (
            config.get("type") != "clt"
            or not all(
                isinstance(config.get(key), str)
                and config[key].startswith(required_hook)
                for key in ("hf_hook_point_in", "hf_hook_point_out")
            )
        ):
            raise ValueError("Official cross-layer input/output hooks required")
        first = layers[0]["w_enc"]
        if not isinstance(first, torch.Tensor) or first.ndim != 2:
            raise ValueError("Expected original per-layer CLT encoder matrix")
        d_model, features = map(int, first.shape)
        if features * count != int(config.get("width", -1)):
            raise ValueError("CLT total source width differs from official config")
        affine = bool(config.get("affine_connection", False))
        self.sources = nn.ModuleList([
            CLTLayer(
                shard, n_layers=count, n_features=features,
                d_model=d_model, affine=affine,
            ) for shard in layers
        ])
        self.n_layers = count
        self.input_dim = d_model
        self.features_per_layer = features
        self.affine = affine
        self.input_hook_template = config["hf_hook_point_in"]
        self.output_hook_template = config["hf_hook_point_out"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Source activations shaped [layers, batch, d_model]."""
        if x.ndim != 3 or tuple(x.shape[:1]) != (self.n_layers,) or x.shape[-1] != self.input_dim:
            raise ValueError("Native CLT input requires all layer source activations")
        return torch.stack([
            layer.encode(x[i]) for i, layer in enumerate(self.sources)
        ])

    def decode(
        self, codes: torch.Tensor, *, input_acts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum triangular source-to-target decoder vectors and each target bias once."""
        if (
            codes.ndim != 3 or tuple(codes.shape[:1]) != (self.n_layers,)
            or codes.shape[-1] != self.features_per_layer
        ):
            raise ValueError("CLT codes must contain every source layer")
        if self.affine and input_acts is None:
            raise ValueError("Affine CLT decode needs the original residual inputs")
        if input_acts is not None and (
            input_acts.shape != (self.n_layers, codes.shape[1], self.input_dim)
        ):
            raise ValueError("CLT skip inputs do not align all model layers")
        output = torch.stack([
            layer.b_dec.expand(codes.shape[1], -1).clone()
            for layer in self.sources
        ])
        for source, shard in enumerate(self.sources):
            code = codes[source].to(device=shard.w_dec.device, dtype=shard.w_dec.dtype)
            # Original Google w_dec[source] maps [F,L,D].
            addition = torch.einsum("bf,ftd->tbd", code, shard.w_dec)
            output[source:] += addition[source:]
        if self.affine:
            assert input_acts is not None
            for target, shard in enumerate(self.sources):
                output[target] += (
                    input_acts[target].to(
                        device=shard.affine_skip_connection.device,
                        dtype=shard.affine_skip_connection.dtype,
                    ) @ shard.affine_skip_connection
                )
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), input_acts=x)

    def edit_delta(
        self, *, source_layer: int, feature_id: int,
        input_activation: torch.Tensor, new_code: float = 0.0,
    ) -> torch.Tensor:
        """Output-space intervention across all downstream layer sites.

        Returns [n_target_layers, batch, d_model]; only target >= source changes.
        """
        if not 0 <= source_layer < self.n_layers:
            raise IndexError("CLT feature source layer out of range")
        if input_activation.ndim != 2 or input_activation.shape[-1] != self.input_dim:
            raise ValueError("Selected CLT source must have [batch, d_model]")
        return self.sources[source_layer].output_delta(
            input_activation, source_layer=source_layer,
            feature_id=feature_id, new_code=new_code,
        )


def load_official_scope2_clt(
    *,
    repo_id: str,
    revision: str,
    directory: str,
    n_layers: int,
    max_total_bytes: int = 6_000_000_000,
    device: str = "cpu",
) -> tuple[Scope2CLT, CLTIdentity]:
    """Bounded explicit original shard download; no unsafe pickle or wildcard revision."""
    if _validate_source(repo_id, revision, directory) != "clt":
        raise ValueError("Native cross-layer transcoder requires official clt/ folder")
    if not 2 <= n_layers <= 64 or not 0 < max_total_bytes <= 64_000_000_000:
        raise ValueError("CLT layer count and memory budget must be explicit")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = Path(hf_hub_download(
        repo_id=repo_id, filename=directory + "/config.json", revision=revision,
    ))
    if config_path.stat().st_size > 16384:
        raise ValueError("Unbounded original CLT source config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Invalid source config JSON")
    files = [config_path]
    used = config_path.stat().st_size
    for layer_id in range(n_layers):
        path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=f"{directory}/params_layer_{layer_id}.safetensors",
            revision=revision,
        ))
        used += path.stat().st_size
        if used > max_total_bytes:
            raise ValueError("Official CLT shards exceed explicitly approved budget")
        files.append(path)
    layers = [load_file(str(path), device="cpu") for path in files[1:]]
    representation = Scope2CLT(layers, config=config).eval()
    if device != "cpu":
        representation = representation.to(device)
    sha = hashlib.sha256()
    for path in files:
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(data)
    meta = CLTIdentity(
        repo_id=repo_id, revision=revision,
        asset_directory=directory,
        input_hook_template=representation.input_hook_template,
        output_hook_template=representation.output_hook_template,
        model_name=str(config.get("model_name", "")),
        source_sha256=sha.hexdigest(),
        n_layers=n_layers, d_model=representation.input_dim,
        features_per_layer=representation.features_per_layer,
        affine_skip=representation.affine,
    )
    return representation, meta
