"""Fast SAE decoder-direction steering through interp-engine KV-cached generation.

This uses interp-engine's actual steer() + generate_stream() APIs, including
per-request vLLM steering and static CUDA-graph taps where available. Fixed
decoder-direction addition is NOT dynamic tokenwise SAE ablation: use the
separate HF live-SAE cached implementation when the current z_t is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class FastSteeringResult:
    backend: str
    generated_token_ids: tuple[int, ...]
    elapsed_seconds: float
    feature_id: int
    strength: float
    baseline_token_ids: tuple[int, ...] | None = None
    protocol: str = "persistent_decoder_direction_kv_v1"

    @property
    def tokens_per_second(self) -> float:
        return len(self.generated_token_ids) / self.elapsed_seconds if self.elapsed_seconds else 0.0


@torch.no_grad()
def decoder_feature_direction(sae: Any, feature_id: int) -> torch.Tensor:
    """Use the original SAE decoder to obtain the *actual* unit-code direction.

    Refuse per-prompt layer_norm SAEs: their inverse normalization cannot be
    represented by one input-independent decoder direction. For affine decoder
    implementations, decode(e_i)-decode(0) is the exact one-unit edit direction.
    """
    cfg = getattr(sae, "cfg", None)
    if cfg is None or not all(hasattr(cfg, x) for x in ("d_in", "d_sae")):
        raise ValueError("SAE must declare cfg.d_in and cfg.d_sae")
    if getattr(cfg, "normalize_activations", None) not in (None, "none"):
        raise ValueError("per-input normalized SAE requires live edit, not a fixed direction")
    if type(feature_id) is not int or not 0 <= feature_id < int(cfg.d_sae):
        raise ValueError("feature_id outside SAE code width")
    if not callable(getattr(sae, "decode", None)):
        raise TypeError("SAE must expose decode")
    try:
        param = next(sae.parameters())
    except (AttributeError, StopIteration) as exc:
        raise TypeError("SAE must expose an initialized parameter dtype/device") from exc
    zero = torch.zeros((1, int(cfg.d_sae)), dtype=param.dtype, device=param.device)
    one = zero.clone()
    one[0, feature_id] = 1.0
    half = zero.clone()
    half[0, feature_id] = 0.5
    base = sae.decode(zero)
    active = sae.decode(one)
    middle = sae.decode(half)
    if any(
        not isinstance(x, torch.Tensor) or x.shape != (1, int(cfg.d_in))
        for x in (base, active, middle)
    ):
        raise RuntimeError("SAE decode must produce [1, d_in]")
    if not torch.isfinite(base).all() or not torch.isfinite(active).all():
        raise ValueError("decoder direction contains nonfinite values")
    delta = (active - base)[0].detach().float().cpu()
    # The fixed-vector route is only justified for an affine SAE decode. A
    # nonaffine decoder needs actual current z_t, not this approximation.
    if not torch.allclose(
        (middle - base).float(), (active - base).float() * 0.5, rtol=1e-3, atol=2e-3
    ):
        raise ValueError("nonaffine SAE decoder; use live cached feature edits")
    if not delta.any():
        raise ValueError("selected SAE feature has a zero decoder direction")
    return delta


def _validate_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(token_ids)
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("token_ids must be a nonempty sequence of integer IDs")
    return ids


def _require_steer_site(model: Any, layer: int) -> str:
    from interp_engine import Address

    if type(layer) is not int or layer < 0 or layer >= int(model.n_layers):
        raise ValueError("layer outside interp-engine model")
    point = Address("resid_post", layer)
    if callable(getattr(model, "refuses", None)):
        reason = model.refuses(point)
        if reason:
            raise RuntimeError(f"backend cannot access requested SAE site: {reason}")
    if getattr(model, "graph_replay", False):
        if point not in getattr(model, "static_writes", ()):
            raise RuntimeError("static backend lacks predeclared SAE write tap")
    elif not getattr(model, "hooks_available", False):
        raise RuntimeError("generation-only backend cannot perform SAE steering")
    return type(model).__name__


def generate_decoder_direction_cached(
    model: Any,
    sae: Any,
    token_ids: Sequence[int],
    *,
    layer: int,
    feature_id: int,
    strength: float,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 0,
    compare_unsteered: bool = False,
    steer_last_prompt_token: bool = True,
    stop_at_eos: bool = True,
) -> FastSteeringResult:
    """Real cached generation with an SAE feature direction on every decode token.

    Earlier prompt positions are excluded from intervention; the last prompt
    token can be steered to make the first next-token prediction an edit.
    With generated=True every *new* token receives the same direction.
    Previously edited KV are persistent by design (not the legacy protocol).
    """
    from interp_engine.steer import generate_stream, steer
    from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

    ids = _validate_token_ids(token_ids)
    backend = _require_steer_site(model, layer)
    if not isinstance(strength, (float, int)) or not bool(
        torch.isfinite(torch.tensor(float(strength)))
    ):
        raise ValueError("strength must be finite")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens outside supported budget")
    if not isinstance(temperature, (float, int)) or not 0 <= temperature <= 10:
        raise ValueError("temperature outside supported bounds")
    if not isinstance(stop_at_eos, bool):
        raise TypeError("stop_at_eos must be bool")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")

    vector = decoder_feature_direction(sae, feature_id)
    spec = SteeringSpec(
        layers={
            layer: LayerSteeringSpec(
                operations=[AddSpec(vector=vector, scale=float(strength), normalize=False)]
            )
        },
        point="resid_post",
    )
    excluded = list(range(len(ids) - 1)) if steer_last_prompt_token else list(range(len(ids)))

    original: tuple[int, ...] | None = None
    if compare_unsteered:
        original = tuple(
            int(step.token_id)
            for step in generate_stream(
                model,
                ids,
                max_tokens=max_new_tokens,
                temperature=float(temperature),
                seed=seed,
                stop_at_eos=stop_at_eos,
            )
        )
    start = perf_counter()
    with steer(
        model,
        spec,
        prompt_token_ids=ids,
        position_mask=excluded,
        generated=True,
    ):
        generated = tuple(
            int(step.token_id)
            for step in generate_stream(
                model,
                ids,
                max_tokens=max_new_tokens,
                temperature=float(temperature),
                seed=seed,
                stop_at_eos=stop_at_eos,
            )
        )
    if torch.cuda.is_available() and torch.as_tensor(vector).is_cuda:
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    return FastSteeringResult(
        backend=backend,
        generated_token_ids=generated,
        elapsed_seconds=elapsed,
        feature_id=feature_id,
        strength=float(strength),
        baseline_token_ids=original,
    )
