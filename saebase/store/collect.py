"""Collect explicit token sites, then reuse cached activations across SAEs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from saebase.api import ActivationSite, ModelAdapter
from .identity import CodeIdentity, SampleRecord, resolve_positions
from .shards import ActivationWriter, CodeWriter, ShardedReader


@dataclass(frozen=True)
class CollectionStats:
    collected: int
    reused: int
    rows_collected: int


@torch.no_grad()
def capture_selected(model: ModelAdapter, *, tokens: torch.Tensor,
                     site: ActivationSite, positions: tuple[int, ...]) -> torch.Tensor:
    """One model forward per sample for any number of activation positions.

    HF block-output adapter supports a batched offset request; TL uses a
    filtered hook cache. The result has shape [selected_positions, d_model].
    """
    model.capabilities.require("capture")
    ids = model.tokens(tokens=tokens)
    model._assert_site(site, ids)
    if not positions or len(set(positions)) != len(positions) or any(
        type(pos) is not int or pos < 0 or pos >= ids.shape[1] for pos in positions
    ):
        raise ValueError("positions must be distinct valid absolute token indices")
    if model.backend == "transformer_lens":
        _, cache = model.model.run_with_cache(ids, names_filter=site.hook_name)
        if site.hook_name not in cache:
            raise RuntimeError(f"hook {site.hook_name!r} missing from activation cache")
        h = cache[site.hook_name]
        if not isinstance(h, torch.Tensor) or h.ndim != 3 or tuple(h.shape[:2]) != tuple(ids.shape):
            raise RuntimeError("captured tensor must be [1, seq, width]")
        result = h[0, list(positions)]
    else:
        offsets = [ids.shape[1] - 1 - pos for pos in positions]
        result = model.model.get_resid(ids, site.layer, offsets)
    if (not isinstance(result, torch.Tensor) or result.ndim != 2
            or result.shape[0] != len(positions) or result.shape[1] < 1
            or not bool(torch.isfinite(result).all())):
        raise RuntimeError("model capture must return finite [n_positions, width] tensor")
    return result.detach().clone()


@torch.no_grad()
def collect_activations(model: ModelAdapter, samples: Iterable[SampleRecord],
                        writer: ActivationWriter, *, site: ActivationSite) -> CollectionStats:
    """Append only missing samples. A second run makes zero model calls.

    M1's existing ModelAdapter does not forward attention masks to backends.
    Live collection therefore permits unpadded inputs or right-side padding
    only; the causal model's requested positions precede the padding.
    """
    ident = writer.activation_identity
    if ident.hook_name != site.hook_name or ident.layer != site.layer:
        raise ValueError("requested hook/layer differs from activation identity")
    new, reused, rows = 0, 0, 0
    for record in samples:
        if writer.has(record):
            reused += 1
            continue
        mask = record.attention_mask
        if mask is not None:
            # Right-padded causal model inputs do not modify earlier activations;
            # left-padding would need explicit attention-mask backend support.
            seen_padding = False
            for x in mask:
                if x == 0:
                    seen_padding = True
                elif seen_padding:
                    raise ValueError("live model capture requires right-side padding or no padding")
        positions = resolve_positions(record)
        ids = torch.as_tensor(record.token_ids, dtype=torch.long).unsqueeze(0)
        result = capture_selected(model, tokens=ids, site=site, positions=positions)
        if writer.append(record, result):
            new += 1
            rows += len(positions)
    writer.flush()
    return CollectionStats(new, reused, rows)


@torch.no_grad()
def encode_from_store(activation_path: str | Path, sae: Any,
                      identity: CodeIdentity, *, layout: str = "csr",
                      max_rows: int = 256, shard_target_bytes: int = 128 * 1024 * 1024,
                      repair_corruption: bool = False) -> tuple[Path, CollectionStats]:
    """Encode cached activations without a model forward.

    The SAE must have deterministic inference (per-token / fixed-threshold).
    This is not training. Rows are never silently truncated to display top-k.
    """
    reader = ShardedReader(activation_path, verify_shards=True)
    if reader.kind != "activations" or reader.width != identity.d_in:
        raise ValueError("SAE input dimension does not match activation cache")
    cfg = getattr(sae, "cfg", None)
    if cfg is None or int(getattr(cfg, "d_in", -1)) != identity.d_in or int(getattr(cfg, "d_sae", -1)) != identity.d_sae:
        raise ValueError("loaded SAE dimensions differ from CodeIdentity")
    if not callable(getattr(sae, "encode", None)):
        raise TypeError("sae.encode() is required")
    try:
        param = next(sae.parameters())
        device, dtype = param.device, param.dtype
    except (AttributeError, StopIteration, TypeError):
        device, dtype = torch.device("cpu"), torch.float32
    written = reused = rows = 0
    with CodeWriter(activation_path, identity, layout=layout, shard_target_bytes=shard_target_bytes,
                    repair_corruption=repair_corruption) as writer:
        # iter_batches caps intermediate encoded tensor allocations for a group.
        for batch in reader.iter_batches(max_rows=max_rows):
            for sample in batch:
                if writer.has(sample.record):
                    reused += 1
                    continue
                matrix = np.array(sample.values, dtype=np.float32, copy=True)
                inputs = torch.as_tensor(matrix, dtype=dtype, device=device)
                encoded = sae.encode(inputs)
                if (not isinstance(encoded, torch.Tensor) or encoded.shape != (len(sample.positions), identity.d_sae)
                        or not bool(torch.isfinite(encoded).all())):
                    raise RuntimeError("sae.encode must return finite [selected_positions, d_sae]")
                if writer.append(sample.record, encoded):
                    written += 1
                    rows += len(sample.positions)
        path = writer.path
    return path, CollectionStats(written, reused, rows)
