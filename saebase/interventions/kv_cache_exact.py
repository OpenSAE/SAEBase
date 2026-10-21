"""Experimental clean-prefix KV decoding with exact SAEBase edit semantics.

The clean KV is never advanced by an edited forward. Each step runs a clean
one-token decode and an independent edited one-token decode against the SAME
clean history. This is an independent implementation to verify against the
existing full-prefix reference, not a parity-verified backend by default.

The cache fork currently uses deepcopy for correctness and bounded tests; it
may not accelerate long prefixes until a non-mutating cache fork is validated.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from saebase.api import ActivationSite
from saebase.api.features import FeatureSnapshot, snapshot_from_activation
from .operators import apply_condition
from .specs import Condition


class KVParityError(RuntimeError):
    """The candidate changed the original stepwise SAE feature protocol."""


@dataclass(frozen=True)
class CleanKVStep:
    clean_logits: torch.Tensor
    edited_logits: torch.Tensor
    snapshot: FeatureSnapshot
    patch_activation: torch.Tensor
    prefix_length: int


class CleanPrefixKVDecoder:
    """One owner, one sequence, Qwen3-compatible HF causal LM only.

    Model must remain in eval mode with fixed weights. Never share an instance
    between sequences or between threads. A separate instance is required for
    baseline and steered token histories once the continuations diverge.
    """

    def __init__(self, wrapped_model: Any, sae: Any, *, layer: int, hook_name: str) -> None:
        raw = getattr(wrapped_model, "_model", None)
        cfg = getattr(raw, "config", None)
        if raw is None or getattr(cfg, "model_type", None) != "qwen3":
            raise ValueError("clean-prefix KV prototype currently supports Qwen3 HF only")
        if getattr(raw, "training", True):
            raise ValueError("clean-prefix KV requires model.eval()")
        if not 0 <= layer < int(wrapped_model.n_layers):
            raise ValueError("layer outside model")
        if hook_name != f"blocks.{layer}.hook_resid_post":
            raise ValueError("only matching Qwen3 resid_post is supported")
        if not callable(getattr(sae, "encode", None)):
            raise TypeError("SAE encoder is required")
        self.model = wrapped_model
        self.raw = raw
        self.sae = sae
        self.layer = layer
        self.hook_name = hook_name
        self.ids: torch.Tensor | None = None
        self.clean_past: Any = None
        self._closed = False

    @torch.no_grad()
    def start(self, prompt_ids: torch.Tensor) -> None:
        if self.ids is not None:
            raise RuntimeError("each decoder may be started only once")
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
            raise ValueError("prompt token IDs must have shape [1, >=1]")
        ids = prompt_ids.detach().clone().to(self.model.cfg.device)
        self.ids = ids
        if ids.shape[1] > 1:
            length = ids.shape[1] - 1
            prefix = ids[:, :length]
            output = self.raw(
                input_ids=prefix, use_cache=True,
                attention_mask=torch.ones_like(prefix),
                cache_position=torch.arange(length, device=ids.device),
            )
            self.clean_past = output.past_key_values
            if self.clean_past is None or self.clean_past.get_seq_length() != length:
                raise RuntimeError("Qwen3 did not return a correctly sized clean KV cache")

    @torch.no_grad()
    def step(self, condition: Condition) -> CleanKVStep:
        if self._closed or self.ids is None:
            raise RuntimeError("decoder must be started and open")
        ids = self.ids
        n = int(ids.shape[1])
        past = self.clean_past
        if past is not None and int(past.get_seq_length()) != n - 1:
            raise RuntimeError("clean KV length differs from the token prefix")
        # A patch may update past_key_values even with use_cache=False.
        # Fork BEFORE either step. Never pass the clean cache into the edit.
        edited_past = copy.deepcopy(past)
        current_token = ids[:, -1:]
        kwargs = {
            "input_ids": current_token,
            "use_cache": True,
            "attention_mask": torch.ones((1, n), dtype=torch.long, device=ids.device),
            "cache_position": torch.tensor([n - 1], dtype=torch.long, device=ids.device),
        }
        layer = self.model._layer_module(self.layer)
        captured: dict[str, torch.Tensor] = {}

        def capture(_module: Any, _inp: Any, out: Any) -> Any:
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
                raise KVParityError("incremental capture shape mismatch")
            captured["h"] = hidden[0, -1].detach().clone()
            return out

        hook = layer.register_forward_hook(capture)
        try:
            clean = self.raw(**kwargs, past_key_values=past)
        finally:
            hook.remove()
        if "h" not in captured or clean.past_key_values is None:
            raise KVParityError("clean incremental forward did not capture/cache")
        if int(clean.past_key_values.get_seq_length()) != n:
            raise KVParityError("clean KV was not advanced by exactly one token")
        site = ActivationSite(self.layer, self.hook_name, n - 1)
        snapshot = snapshot_from_activation(
            self.sae, tokens=ids, site=site, activation=captured["h"]
        )
        patch = apply_condition(self.sae, snapshot, condition)

        def replace(_module: Any, _inp: Any, out: Any) -> Any:
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
                raise KVParityError("incremental patch shape mismatch")
            changed = hidden.clone()
            changed[0, -1] = patch.activation.to(device=hidden.device, dtype=hidden.dtype)
            return (changed, *out[1:]) if isinstance(out, tuple) else changed

        hook = layer.register_forward_hook(replace)
        try:
            edited = self.raw(**kwargs, past_key_values=edited_past)
        finally:
            hook.remove()
        # The edited cache is discarded. The next step sees ONLY clean history.
        self.clean_past = clean.past_key_values
        return CleanKVStep(
            clean_logits=clean.logits[0, -1].detach().clone(),
            edited_logits=edited.logits[0, -1].detach().clone(),
            snapshot=snapshot,
            patch_activation=patch.activation.detach().clone(),
            prefix_length=n,
        )

    def append(self, next_token: int) -> None:
        if self._closed or self.ids is None or self.clean_past is None:
            raise RuntimeError("run step() before append()")
        if not isinstance(next_token, int) or next_token < 0:
            raise ValueError("next token must be a nonnegative ID")
        self.ids = torch.cat(
            (self.ids, torch.tensor([[next_token]], device=self.ids.device)), dim=1
        )

    def close(self) -> None:
        self.clean_past = None
        self.ids = None
        self._closed = True


@torch.no_grad()
def compare_cached_step_with_reference(
    decoder: CleanPrefixKVDecoder, adapter: Any, sae: Any, condition: Condition,
    *, atol: float, rtol: float,
) -> dict[str, Any]:
    """Fail closed on mismatched features or logits before accepting a cached step."""
    from saebase.api import capture_features

    if decoder.ids is None:
        raise RuntimeError("decoder is not started")
    ids = decoder.ids
    site = ActivationSite(decoder.layer, decoder.hook_name, ids.shape[1] - 1)
    reference_snapshot = capture_features(adapter, sae, site=site, tokens=ids)
    reference_patch = apply_condition(sae, reference_snapshot, condition)
    baseline = adapter.forward_logits(tokens=ids)
    edited = adapter.patch_logits(
        tokens=ids, site=site, activation=reference_patch.activation
    )
    candidate = decoder.step(condition)
    ref_support = reference_snapshot.codes != 0
    new_support = candidate.snapshot.codes != 0
    flip_count = int((ref_support != new_support).sum().item())
    errors = {
        "activation": float((reference_snapshot.activation.float() -
                              candidate.snapshot.activation.float()).abs().max().item()),
        "clean_logits": float((baseline.float() - candidate.clean_logits.float()).abs().max().item()),
        "edited_logits": float((edited.float() - candidate.edited_logits.float()).abs().max().item()),
    }
    passed = (
        flip_count == 0
        and torch.allclose(reference_snapshot.activation.float(),
                           candidate.snapshot.activation.float(), atol=atol, rtol=rtol)
        and torch.allclose(baseline.float(), candidate.clean_logits.float(),
                           atol=atol, rtol=rtol)
        and torch.allclose(edited.float(), candidate.edited_logits.float(),
                           atol=atol, rtol=rtol)
    )
    result = {
        "pass": bool(passed), "feature_flips": flip_count, "max_abs": errors,
        "prefix_length": candidate.prefix_length,
    }
    if not passed:
        raise KVParityError(f"incremental KV differs from full-prefix oracle: {result}")
    return result
