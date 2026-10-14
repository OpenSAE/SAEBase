"""Runtime contracts for prompt artifacts and SAE activation health."""

from __future__ import annotations

from typing import Any

import torch

from saebase.data.artifact_layout import canonicalize_behavior


def assert_manifest_matches(
    manifest: dict[str, Any],
    *,
    behavior: str,
    model_key: str,
) -> None:
    beh = canonicalize_behavior(behavior)
    if str(manifest.get("model_key")) != str(model_key):
        raise RuntimeError(
            f"manifest.model_key={manifest.get('model_key')!r} != requested={model_key!r}"
        )
    man_beh = canonicalize_behavior(str(manifest.get("behavior", beh)))
    if man_beh != beh:
        raise RuntimeError(
            f"manifest.behavior={man_beh!r} != requested={beh!r}"
        )


def _vocab_size(model: Any) -> int | None:
    for attr in ("d_vocab", "d_vocab_out"):
        cfg = getattr(model, "cfg", None)
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    tok = getattr(model, "tokenizer", None)
    if tok is not None and hasattr(tok, "vocab_size"):
        return int(tok.vocab_size)
    return None


def assert_prompt_record(
    ex: dict[str, Any],
    *,
    model: Any,
    ctx: str,
) -> None:
    if "prompt" not in ex or not str(ex["prompt"]).strip():
        raise ValueError(f"{ctx}: missing non-empty prompt")

    token_ids: list[int] = []
    tok = ex.get("tokenization")
    if isinstance(tok, dict):
        for key in (
            "correct_ids",
            "distractor_ids",
            "greater_ids",
            "less_equal_ids",
            "positive_ids",
            "negative_ids",
            "input_ids",
        ):
            vals = tok.get(key)
            if isinstance(vals, list):
                token_ids.extend(int(v) for v in vals)
    for key in (
        "tok_correct",
        "tok_distractor",
        "tok_true",
        "tok_false",
        "tok_io",
        "tok_s",
    ):
        if key in ex:
            token_ids.append(int(ex[key]))
    for key in ("greater_tok_ids", "less_equal_tok_ids"):
        if key in ex and isinstance(ex[key], list):
            token_ids.extend(int(v) for v in ex[key])

    if not token_ids:
        raise ValueError(f"{ctx}: missing tokenization / token id fields")

    # For vocab checks, ignore long input_ids and focus on target ids when present.
    check_ids = token_ids
    if isinstance(tok, dict) and (
        "positive_ids" in tok or "correct_ids" in tok or "tok_correct" in ex
    ):
        check_ids = [
            t
            for t in token_ids
            if not (
                isinstance(tok.get("input_ids"), list)
                and t in set(int(x) for x in tok.get("input_ids", []))
                and t
                not in set(
                    int(x)
                    for key in ("positive_ids", "negative_ids", "correct_ids", "distractor_ids")
                    for x in (tok.get(key) or [])
                )
            )
        ] or token_ids

    vocab = _vocab_size(model)
    if vocab is not None:
        bad = [t for t in check_ids if t < 0 or t >= vocab]
        if bad:
            raise ValueError(
                f"{ctx}: token ids out of vocab range [0, {vocab}): {bad[:5]}"
            )


def assert_activation_sae_shapes(
    h: torch.Tensor,
    z: torch.Tensor,
    sae: Any,
    *,
    ctx: str,
) -> None:
    d_in = int(sae.cfg.d_in)
    d_sae = int(sae.cfg.d_sae)
    if h.ndim != 1:
        raise ValueError(f"{ctx}: expected last-token activation 1-D, got {tuple(h.shape)}")
    if int(h.shape[-1]) != d_in:
        raise RuntimeError(
            f"{ctx}: activation dim {int(h.shape[-1])} != sae.d_in {d_in}"
        )
    if z.ndim != 1:
        raise ValueError(f"{ctx}: expected latent 1-D, got {tuple(z.shape)}")
    if int(z.shape[-1]) != d_sae:
        raise RuntimeError(
            f"{ctx}: latent dim {int(z.shape[-1])} != sae.d_sae {d_sae}"
        )


def assert_finite_tensors(*tensors: torch.Tensor, ctx: str) -> None:
    for i, t in enumerate(tensors):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"{ctx}: non-finite values in tensor #{i}")


def latent_l0(z: torch.Tensor) -> int:
    return int((z != 0).sum().item())


def reconstruction_stats(h: torch.Tensor, h_hat: torch.Tensor) -> dict[str, float]:
    mse = float(((h - h_hat) ** 2).mean().item())
    denom = float((h**2).mean().item()) + 1e-8
    # Fraction of variance explained around the activation mean.
    var = float(((h - h.mean()) ** 2).sum().item()) + 1e-12
    err = float(((h - h_hat) ** 2).sum().item())
    fve = 1.0 - err / var
    return {
        "reconstruction_mse": mse,
        "relative_reconstruction_error": mse / denom,
        "fve": float(fve),
    }
