"""Backend-neutral, one-prompt activation and model-intervention interface.

Capture/patch are deliberately scoped to an explicit model site and a single
prompt. This contract prevents ambiguity about batch and token alignment.
Supported adapters: local TransformerLens HookedTransformer/TransformerBridge, HFHooksModel and
compatible implementations of their documented methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class UnsupportedCapability(RuntimeError):
    """The selected backend cannot perform the requested operation."""


@dataclass(frozen=True)
class ActivationSite:
    """A model hook location and a Python-style token index (-1 is last).

    ``layer`` is the zero-based transformer block index and must agree with
    a hook like ``blocks.3.hook_resid_post``. The specific hook kind is part
    of the identity; an HF layer-output hook only supports resid_post.
    """

    layer: int
    hook_name: str
    position: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not isinstance(self.hook_name, str) or not self.hook_name.strip():
            raise ValueError("hook_name must be a non-empty string")
        if isinstance(self.position, bool) or not isinstance(self.position, int):
            raise ValueError("position must be an integer")
        pieces = self.hook_name.split(".")
        if len(pieces) >= 3 and pieces[0] == "blocks" and pieces[1].isdigit():
            if int(pieces[1]) != self.layer:
                raise ValueError(
                    f"hook {self.hook_name!r} belongs to layer {pieces[1]}, not layer {self.layer}"
                )


@dataclass(frozen=True)
class ModelCapabilities:
    forward: bool
    capture: bool
    patch: bool
    generate: bool
    backend: str

    def require(self, capability: str) -> None:
        if capability not in ("forward", "capture", "patch", "generate"):
            raise ValueError(f"Unknown model capability: {capability}")
        if not getattr(self, capability):
            raise UnsupportedCapability(
                f"{self.backend} backend cannot {capability}; use an adapter "
                "that supplies this capability"
            )


def _index(length: int, position: int) -> int:
    if isinstance(position, bool) or not isinstance(position, int):
        raise TypeError("position must be an integer")
    if length < 1 or position < -length or position >= length:
        raise IndexError(f"token position {position} outside sequence of length {length}")
    return position % length


def _as_logits(output: Any, *, token_index: int, length: int) -> torch.Tensor:
    tensor = getattr(output, "logits", output)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError("model forward must return logits with shape [batch, seq, vocab]")
    if tensor.shape[0] != 1 or tensor.shape[1] != length:
        raise ValueError(
            f"logits shape {tuple(tensor.shape)} incompatible with [1, {length}, vocab]"
        )
    return tensor[0, token_index].detach().clone()


class ModelAdapter:
    """Wrap a loaded model with a tested, explicit activation interface.

    Uses existing `run_with_cache`/`run_with_hooks` for TransformerLens or
    `get_resid`/`run_with_patch` for the SAEBase HF residual backend.
    Logit-only models can still be queried without advertising patch support.
    """

    def __init__(self, model: Any, *, backend: str = "auto") -> None:
        self.model = model
        if backend == "auto":
            if getattr(model, "_saebase_backend", None) == "hf_hooks":
                backend = "hf_hooks"
            elif getattr(model, "_saebase_backend", None) == "hf":
                backend = "logits_only"
            elif callable(getattr(model, "run_with_cache", None)) or callable(
                getattr(model, "run_with_hooks", None)
            ):
                backend = "transformer_lens"
            elif callable(getattr(model, "get_resid", None)) or callable(
                getattr(model, "run_with_patch", None)
            ):
                backend = "hf_hooks"
            else:
                backend = "logits_only"
        if backend not in {"transformer_lens", "hf_hooks", "logits_only"}:
            raise ValueError(f"Unknown backend: {backend}")
        capture_method = {"transformer_lens": "run_with_cache", "hf_hooks": "get_resid"}.get(
            backend
        )
        patch_method = {"transformer_lens": "run_with_hooks", "hf_hooks": "run_with_patch"}.get(
            backend
        )
        can_capture = bool(capture_method and callable(getattr(model, capture_method, None)))
        can_patch = bool(patch_method and callable(getattr(model, patch_method, None)))
        if backend != "logits_only" and not (can_capture or can_patch):
            raise TypeError(f"{backend} model does not expose activation methods")
        self.backend = backend
        self.capabilities = ModelCapabilities(
            forward=callable(model),
            capture=can_capture,
            patch=can_patch,
            generate=callable(getattr(model, "generate", None)),
            backend=backend,
        )

    def available_hooks(self, *, layer: int | None = None) -> tuple[str, ...]:
        """List actual hook names without running the model.

        Alias and driver coverage depend on the loaded backend. Empty output
        means hook enumeration is unsupported, not that every name is valid.
        """
        if self.backend != "transformer_lens":
            return ()
        registry = getattr(self.model, "hook_dict", None)
        if registry is None:
            registry = getattr(self.model, "hook_points", None)
        if isinstance(registry, dict):
            names = tuple(str(name) for name in registry)
        elif isinstance(registry, (tuple, list, set)):
            names = tuple(str(name) for name in registry)
        else:
            return ()
        if layer is not None:
            if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                raise ValueError("layer must be a nonnegative integer")
            prefix = f"blocks.{layer}."
            names = tuple(name for name in names if name.startswith(prefix))
        return tuple(sorted(set(names)))

    def _tokens(self, *, prompt: str | None, tokens: torch.Tensor | None) -> torch.Tensor:
        if (prompt is None) == (tokens is None):
            raise ValueError("provide exactly one of prompt or tokens")
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a non-empty string")
            convert = getattr(self.model, "to_tokens", None)
            if not callable(convert):
                raise UnsupportedCapability("model requires to_tokens() for a string prompt")
            tokens = convert(prompt)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have shape [1, length] with length >= 1")
        if tokens.dtype not in {torch.int32, torch.int64}:
            raise TypeError("tokens must contain integer token IDs")
        # Keep tokens aligned with the model even if a caller supplied CPU ids.
        try:
            device = next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            cfg = getattr(self.model, "cfg", None)
            device = getattr(cfg, "device", tokens.device)
        return tokens.to(device)

    def tokens(
        self, *, prompt: str | None = None, tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._tokens(prompt=prompt, tokens=tokens)

    def _assert_site(self, site: ActivationSite, tokens: torch.Tensor) -> int:
        if not isinstance(site, ActivationSite):
            raise TypeError("site must be an ActivationSite")
        pos = _index(tokens.shape[1], site.position)
        if self.backend == "hf_hooks" and site.hook_name != f"blocks.{site.layer}.hook_resid_post":
            raise UnsupportedCapability(
                "HFHooksModel captures the transformer block output only; "
                f"requested site {site.hook_name!r}"
            )
        cfg = getattr(self.model, "cfg", None)
        layers = getattr(cfg, "n_layers", None)
        if layers is not None and not (0 <= site.layer < int(layers)):
            raise ValueError(f"layer {site.layer} outside model's {layers} layers")
        if self.backend == "transformer_lens":
            # A names_filter that matches no hooks is not an error in
            # TransformerLens. Reject unknown names *before* patch execution.
            get_point = getattr(self.model, "get_hook_point", None)
            if callable(get_point) and get_point(site.hook_name) is None:
                raise UnsupportedCapability(
                    f"model has no fireable hook {site.hook_name!r}"
                )
        return pos

    @torch.no_grad()
    def forward_logits(self, *, tokens: torch.Tensor, readout_position: int = -1) -> torch.Tensor:
        self.capabilities.require("forward")
        ids = self._tokens(prompt=None, tokens=tokens)
        pos = _index(ids.shape[1], readout_position)
        return _as_logits(self.model(ids), token_index=pos, length=ids.shape[1])

    @torch.no_grad()
    def capture(self, *, tokens: torch.Tensor, site: ActivationSite) -> torch.Tensor:
        self.capabilities.require("capture")
        ids = self._tokens(prompt=None, tokens=tokens)
        pos = self._assert_site(site, ids)
        if self.backend == "transformer_lens":
            _, cache = self.model.run_with_cache(ids, names_filter=site.hook_name)
            if site.hook_name not in cache:
                raise RuntimeError(f"hook {site.hook_name!r} missing from model cache")
            activation = cache[site.hook_name]
            if not isinstance(activation, torch.Tensor) or activation.ndim != 3:
                raise RuntimeError("expected hook activation shape [batch, seq, d_model]")
            if activation.shape[:2] != ids.shape:
                raise RuntimeError("hook activation batch/sequence does not match tokens")
            out = activation[0, pos]
        else:
            offset = int(ids.shape[1] - 1 - pos)
            h = self.model.get_resid(ids, site.layer, [offset])
            if not isinstance(h, torch.Tensor) or h.ndim != 2 or h.shape[0] != 1:
                raise RuntimeError("get_resid must return shape [n_offsets, d_model]")
            out = h[0]
        if out.ndim != 1 or not bool(torch.isfinite(out).all()):
            raise RuntimeError("captured activation must be finite and one-dimensional")
        return out.detach().clone()

    def supports_capture_with_logits(self) -> bool:
        """True when one forward can return the site activation and readout logits."""
        return callable(getattr(self.model, "capture_with_logits", None))

    @torch.no_grad()
    def capture_with_logits(
        self, *, tokens: torch.Tensor, site: ActivationSite, readout_position: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Activation at the site and baseline logits from a single forward when supported."""
        if not self.supports_capture_with_logits():
            return (
                self.capture(tokens=tokens, site=site),
                self.forward_logits(tokens=tokens, readout_position=readout_position),
            )
        self.capabilities.require("capture")
        self.capabilities.require("forward")
        ids = self._tokens(prompt=None, tokens=tokens)
        pos = self._assert_site(site, ids)
        read = _index(ids.shape[1], readout_position)
        if self.backend != "hf_hooks":
            return (
                self.capture(tokens=ids, site=site),
                self.forward_logits(tokens=ids, readout_position=readout_position),
            )
        offset = int(ids.shape[1] - 1 - pos)
        read_offset = int(ids.shape[1] - 1 - read)
        captured = self.model.capture_with_logits(
            ids, site.layer, [offset], readout_offset=read_offset
        )
        if not isinstance(captured, tuple) or len(captured) != 2:
            raise RuntimeError("capture_with_logits must return (activation, logits)")
        activation, logits = captured
        if (
            not isinstance(activation, torch.Tensor)
            or activation.ndim != 2
            or activation.shape[0] != 1
        ):
            raise RuntimeError("capture_with_logits activation must have shape [1, d_model]")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
            raise RuntimeError("capture_with_logits logits must have shape [vocab]")
        if not bool(torch.isfinite(activation).all()) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("capture_with_logits returned a non-finite tensor")
        return activation[0].detach().clone(), logits.detach().clone()

    @torch.no_grad()
    def patch_logits(
        self,
        *,
        tokens: torch.Tensor,
        site: ActivationSite,
        activation: torch.Tensor,
        readout_position: int = -1,
    ) -> torch.Tensor:
        self.capabilities.require("patch")
        ids = self._tokens(prompt=None, tokens=tokens)
        pos = self._assert_site(site, ids)
        readout = _index(ids.shape[1], readout_position)
        if not isinstance(activation, torch.Tensor) or activation.ndim != 1:
            raise ValueError("patch activation must be a vector with shape [d_model]")
        if not bool(torch.isfinite(activation).all()):
            raise ValueError("patch activation contains non-finite values")
        if self.backend == "hf_hooks":
            offset = ids.shape[1] - 1 - pos
            result = self.model.run_with_patch(
                ids,
                site.layer,
                {offset: activation},
                readout_offset=ids.shape[1] - 1 - readout,
            )
            if not isinstance(result, torch.Tensor) or result.ndim != 1:
                raise RuntimeError("HFHooksModel patch must return logits of shape [vocab]")
            return result.detach().clone()

        # TransformerLens 4 may pass the hook by keyword; earlier versions
        # pass it positionally. The fired counter rejects a silent no-op when
        # an in-range hook alias exists but is not actually invoked.
        fired = 0

        def replace(h: torch.Tensor, hook: Any = None, **kwargs: Any) -> torch.Tensor:
            nonlocal fired
            fired += 1
            if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[:2] != ids.shape:
                raise RuntimeError("patch hook must receive shape [1, seq, d_model]")
            if h.shape[-1] != activation.numel():
                raise ValueError(f"patch dim {activation.numel()} != model dim {h.shape[-1]}")
            patched = h.clone()
            patched[0, pos] = activation.to(device=h.device, dtype=h.dtype)
            return patched

        output = self.model.run_with_hooks(ids, fwd_hooks=[(site.hook_name, replace)])
        if fired != 1:
            raise UnsupportedCapability(
                f"hook {site.hook_name!r} fired {fired} times; expected exactly once"
            )
        return _as_logits(output, token_index=readout, length=ids.shape[1])

    @torch.no_grad()
    def assert_identity_patch(
        self, *, tokens: torch.Tensor, site: ActivationSite, rtol: float = 1e-4, atol: float = 1e-4
    ) -> None:
        """Original activation patch must reproduce the unmodified forward logits."""
        original = self.capture(tokens=tokens, site=site)
        actual = self.patch_logits(tokens=tokens, site=site, activation=original)
        expected = self.forward_logits(tokens=tokens)
        if not torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol):
            diff = float((actual.float() - expected.float()).abs().max().item())
            raise RuntimeError(
                f"identity patch changes model output (max absolute difference {diff:.6g}); "
                "verify site, model eval mode and patch backend"
            )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to a backend-provided generator when available."""
        self.capabilities.require("generate")
        return self.model.generate(*args, **kwargs)
