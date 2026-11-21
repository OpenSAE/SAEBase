"""Same-shape residual replay for independent edits on immutable model prefixes.

Supports TransformerLens pre/post residual sites and the existing HF decoder
suffix adapter. Callers must validate numerical equivalence before enabling it.
"""

import inspect
import torch
from saebase.api.model import UnsupportedCapability


class SiteReplay:
    @torch.inference_mode()
    def __init__(self, adapter, tokens, site, *, cache=True, prepared_hidden=None):
        self.adapter, self.raw, self.tokens, self.site = adapter, adapter.model, tokens, site
        self.hidden, self.start = None, None
        if cache and adapter.backend == "transformer_lens":
            if {"start_at_layer", "stop_at_layer"}.issubset(
                inspect.signature(self.raw.forward).parameters
            ):
                if site.hook_name == f"blocks.{site.layer}.hook_resid_post":
                    self.start = site.layer + 1
                elif site.hook_name == f"blocks.{site.layer}.hook_resid_pre":
                    self.start = site.layer
                if self.start is not None:
                    self.hidden = (
                        prepared_hidden
                        if prepared_hidden is not None
                        else self.raw(tokens, stop_at_layer=self.start).detach()
                    )
        elif (
            cache
            and adapter.backend == "hf_hooks"
            and site.hook_name == f"blocks.{site.layer}.hook_resid_post"
        ):
            if self.raw.supports_suffix_replay():
                self.hidden = (
                    prepared_hidden
                    if prepared_hidden is not None
                    else self.raw.cache_site_hidden(tokens, site.layer)
                )
        self.execution = "cached_suffix" if self.hidden is not None else "full_sequence"

    @torch.inference_mode()
    def logits(self, activations):
        if activations.ndim != 2:
            raise ValueError("Expected [independent edits, hidden dimension]")
        batch = len(activations)
        ids = self.tokens.expand(batch, -1)
        position = self.site.position % self.tokens.shape[1]
        if self.hidden is not None:
            hidden = self.hidden.expand(batch, -1, -1).clone()
            hidden[:, position] = activations.to(hidden)
            if self.adapter.backend == "transformer_lens":
                return self.raw(hidden, start_at_layer=self.start)
            return self.raw.all_logits_from_site_hidden(ids, self.site.layer, hidden)
        if self.adapter.backend == "transformer_lens":

            def patch(hidden, hook=None):
                value = hidden.clone()
                value[:, position] = activations.to(hidden)
                return value

            return self.raw.run_with_hooks(ids, fwd_hooks=[(self.site.hook_name, patch)])
        if self.adapter.backend != "hf_hooks":
            raise UnsupportedCapability(
                "Batched residual replay requires a registered TL or HF decoder"
            )

        def patch(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            value = hidden.clone()
            value[:, position] = activations.to(hidden)
            return (value, *output[1:]) if isinstance(output, tuple) else value

        with self.raw._forward_lock:
            handle = self.raw._layer_module(self.site.layer).register_forward_hook(patch)
            try:
                return self.raw._model(input_ids=ids).logits
            finally:
                handle.remove()


@torch.inference_mode()
def observe_batch(adapter, tokens, site):
    """One real batched forward returns native residuals and clean last logits."""
    if adapter.backend == "transformer_lens":
        logits, cache = adapter.model.run_with_cache(tokens, names_filter=site.hook_name)
        return cache[site.hook_name][:, -1].detach().clone(), logits[:, -1].detach().clone()
    if adapter.backend != "hf_hooks":
        raise UnsupportedCapability("Native observation batches require TL or HF hooks")
    raw = adapter.model
    saved = {}

    def capture(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        saved["hidden"] = value[:, -1].detach().clone()

    with raw._forward_lock:
        handle = raw._layer_module(site.layer).register_forward_hook(capture)
        try:
            logits = raw._model(input_ids=tokens).logits
        finally:
            handle.remove()
    return saved["hidden"], logits[:, -1].detach().clone()


@torch.inference_mode()
def patch_batch(adapter, tokens, site, activations):
    """Patch one absolute position in each independent full-prefix row."""
    position = site.position % tokens.shape[1]

    def patch(hidden, hook=None):
        value = hidden.clone()
        value[:, position] = activations.to(hidden)
        return value

    if adapter.backend == "transformer_lens":
        return (
            adapter.model.run_with_hooks(tokens, fwd_hooks=[(site.hook_name, patch)])[:, -1]
            .detach()
            .clone()
        )
    if adapter.backend != "hf_hooks":
        raise UnsupportedCapability("Native patch batches require TL or HF hooks")
    raw = adapter.model

    def hf_patch(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        value = patch(hidden)
        return (value, *output[1:]) if isinstance(output, tuple) else value

    with raw._forward_lock:
        handle = raw._layer_module(site.layer).register_forward_hook(hf_patch)
        try:
            return raw._model(input_ids=tokens).logits[:, -1].detach().clone()
        finally:
            handle.remove()
