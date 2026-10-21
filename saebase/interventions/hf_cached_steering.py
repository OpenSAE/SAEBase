"""Incremental KV-cached generation with live SAE edits at every new token.

Unlike fixed decoder-direction steering, this implementation encodes the
actual residual at each token, applies SAEBase's original Condition, and then
propagates the edited activation through the remaining model. Past KV from
the edited execution is retained. This is the distinct protocol
persistent_live_sae_kv_v1, NOT the old clean-full-prefix reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import torch

from saebase.api import ActivationSite
from saebase.api.features import snapshot_from_activation
from .operators import apply_condition
from .specs import Condition


@dataclass(frozen=True)
class CachedSAEStep:
    position: int
    next_token: int
    edited_feature_ids: tuple[int, ...]
    edit_l2: float


@dataclass(frozen=True)
class CachedSAEGeneration:
    generated_token_ids: tuple[int, ...]
    elapsed_seconds: float
    model_forwards: int
    steps: tuple[CachedSAEStep, ...]
    protocol: str = "persistent_live_sae_kv_v1"

    @property
    def tokens_per_second(self) -> float:
        return len(self.generated_token_ids) / self.elapsed_seconds if self.elapsed_seconds else 0.0


@torch.no_grad()
def generate_live_sae_cached(
    wrapped_model: Any,
    sae: Any,
    prompt_ids: torch.Tensor,
    *,
    layer: int,
    condition: Condition,
    max_new_tokens: int = 32,
    eos_token_ids: Sequence[int] = (),
    steer_last_prompt_token: bool = True,
    record_steps: bool = False,
    max_context_tokens: int | None = None,
) -> CachedSAEGeneration:
    """One prefill + one cached forward per generated token, live SAE edit at every step.

    The activation hook computes the actual SAE codes at each step.
    This retains edited downstream KV, unlike the historical clean-prefix
    experiment. Execution holds the wrapper's RLock for its entire lifetime.
    Never call the raw underlying model concurrently outside this wrapper.
    """
    raw = getattr(wrapped_model, "_model", None)
    cfg = getattr(raw, "config", None)
    if raw is None or getattr(cfg, "model_type", None) != "qwen3":
        raise ValueError("live cached SAE generation currently requires a Qwen3 HF model")
    if bool(getattr(raw, "training", True)):
        raise ValueError("model must be in eval mode")
    if not callable(getattr(wrapped_model, "_layer_module", None)):
        raise TypeError("HF residual-hook wrapper required")
    if not isinstance(condition, Condition):
        raise TypeError("condition must be a SAEBase Condition")
    if (
        not isinstance(prompt_ids, torch.Tensor)
        or prompt_ids.ndim != 2
        or prompt_ids.shape[0] != 1
        or prompt_ids.shape[1] < 1
        or prompt_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("prompt_ids must be nonempty integer [1, seq] tensor")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if type(layer) is not int or not 0 <= layer < int(wrapped_model.n_layers):
        raise ValueError("layer outside model")
    if any(type(x) is not int or x < 0 for x in eos_token_ids):
        raise ValueError("eos_token_ids must be nonnegative integers")
    if max_context_tokens is not None and (
        type(max_context_tokens) is not int
        or prompt_ids.shape[1] + max_new_tokens > max_context_tokens
    ):
        raise ValueError("requested generation exceeds configured context")
    lock = getattr(wrapped_model, "_forward_lock", None)
    if lock is None:
        raise TypeError("HF wrapper must expose its exclusive forward lock")

    ids = prompt_ids.to(device=wrapped_model.cfg.device, dtype=torch.long).detach()
    site_name = f"blocks.{layer}.hook_resid_post"
    module = wrapped_model._layer_module(layer)
    total = int(ids.shape[1])
    next_ids: list[int] = []
    trace: list[CachedSAEStep] = []
    past: Any = None
    forwards = 0
    capture: dict[str, Any] = {}
    first_prompt = True

    def live_edit(_module: Any, _args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise RuntimeError("SAE hook expected [1, sequence, width] residual")
        if hidden.shape[1] < 1:
            raise RuntimeError("SAE hook received an empty decode sequence")
        if first_prompt and not steer_last_prompt_token:
            return output
        position = total - 1
        vector = hidden[0, -1].detach().clone()
        snapshot = snapshot_from_activation(
            sae,
            tokens=ids,
            site=ActivationSite(layer, site_name, position),
            activation=vector,
        )
        patch = apply_condition(sae, snapshot, condition)
        delta = patch.activation.to(device=vector.device, dtype=vector.dtype) - vector
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError("live SAE patch became nonfinite")
        changed = hidden.clone()
        changed[0, -1] = patch.activation.to(device=hidden.device, dtype=hidden.dtype)
        if record_steps:
            capture["features"] = tuple(
                int(x) for x in torch.nonzero(snapshot.codes != 0).flatten().tolist()
            )
            capture["delta"] = float(torch.linalg.vector_norm(delta.float()).item())
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    start = perf_counter()
    with lock:
        hook = module.register_forward_hook(live_edit)
        try:
            for step in range(max_new_tokens):
                if past is None:
                    position = torch.arange(total, dtype=torch.long, device=ids.device)
                    forward_ids = ids
                else:
                    position = torch.tensor([total - 1], dtype=torch.long, device=ids.device)
                    forward_ids = ids[:, -1:]
                result = raw(
                    input_ids=forward_ids,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=position,
                    attention_mask=torch.ones((1, total), dtype=torch.long, device=ids.device),
                )
                forwards += 1
                past = result.past_key_values
                if past is None or not callable(getattr(past, "get_seq_length", None)):
                    raise RuntimeError("Qwen3 did not return a supported KV cache")
                if int(past.get_seq_length()) != total:
                    raise RuntimeError("KV length mismatch; cached decode cannot continue")
                logits = result.logits
                if logits.ndim != 3 or logits.shape[:2] != (1, forward_ids.shape[1]):
                    raise RuntimeError("invalid cached decode logits")
                new_id = int(torch.argmax(logits[0, -1].float()).item())
                if record_steps:
                    trace.append(
                        CachedSAEStep(
                            position=total - 1,
                            next_token=new_id,
                            edited_feature_ids=capture.get("features", ()),
                            edit_l2=float(capture.get("delta", 0.0)),
                        )
                    )
                next_ids.append(new_id)
                if new_id in eos_token_ids or step == max_new_tokens - 1:
                    break
                ids = torch.cat(
                    (ids, torch.tensor([[new_id]], device=ids.device, dtype=torch.long)), dim=1
                )
                total += 1
                first_prompt = False
        finally:
            hook.remove()
    if ids.device.type == "cuda":
        torch.cuda.synchronize(ids.device)
    elapsed = perf_counter() - start
    return CachedSAEGeneration(
        generated_token_ids=tuple(next_ids),
        elapsed_seconds=elapsed,
        model_forwards=forwards,
        steps=tuple(trace),
    )
