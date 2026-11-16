"""Immutable causal-prefix reuse for repeated last-prompt-token interventions."""

from __future__ import annotations

import torch

from .model import UnsupportedCapability


class FrozenPrefixAnswerReadout:
    """Keep all earlier prompt tokens fixed; recompute the edited token and answer.

    The upstream frozen cache never incorporates an intervention or answer.
    A caller verifies native numerical agreement before using this optimization.
    """

    @torch.no_grad()
    def __init__(self, adapter, prompt_tokens, answer_tokens, site, *, reference_activation=None):
        if adapter.backend != "transformer_lens":
            raise UnsupportedCapability("Frozen prefix scoring requires TransformerLens")
        try:
            from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
        except ImportError as exc:
            raise UnsupportedCapability("This TransformerLens has no frozen prefix cache") from exc
        self.adapter, self.model, self.site = adapter, adapter.model, site
        self.prompt = adapter.tokens(tokens=prompt_tokens)
        self.answer = adapter.tokens(tokens=answer_tokens)
        if adapter._assert_site(site, self.prompt) != self.prompt.shape[1] - 1:
            raise UnsupportedCapability("Frozen prefix edits the last prompt token only")
        self.suffix = torch.cat((self.prompt[:, -1:], self.answer[:, :-1]), dim=1)
        self.cache = TransformerLensKeyValueCache.init_cache(
            self.model.cfg, self.prompt.device, batch_size=1
        )
        if self.prompt.shape[1] > 1:
            self.model(self.prompt[:, :-1], past_kv_cache=self.cache, return_type=None,
                       attention_mask=torch.ones_like(self.prompt[:, :-1]))
        self.cache.freeze()
        self.reference_activation = reference_activation
        self.scoring_activation = None
        if reference_activation is not None:
            _, captured = self.model.run_with_cache(
                self.suffix,
                names_filter=site.hook_name,
                past_kv_cache=self.cache,
                attention_mask=torch.ones_like(self.suffix),
            )
            self.scoring_activation = captured[site.hook_name][0, 0].detach().clone()

    @torch.no_grad()
    def token_logprobs(self, activation=None):
        hooks = []
        if activation is not None:

            def patch(hidden, hook=None):
                value = hidden.clone()
                target = activation
                if self.reference_activation is not None:
                    delta = activation.float().to(
                        hidden.device
                    ) - self.reference_activation.float().to(hidden.device)
                    target = self.scoring_activation.float() + delta
                value[0, 0] = target.to(device=hidden.device, dtype=hidden.dtype)
                return value

            hooks = [(self.site.hook_name, patch)]
        logits = self.model.run_with_hooks(self.suffix, fwd_hooks=hooks, past_kv_cache=self.cache,
                                           attention_mask=torch.ones_like(self.suffix))
        if logits.shape[:2] != self.suffix.shape:
            raise ValueError("Cached answer logits do not align with answer tokens")
        return logits[0].float().log_softmax(-1).gather(1, self.answer[0, :, None]).squeeze(1)

    @torch.no_grad()
    def verify(self, activations, *, reference_readout=None, atol=1e-4, rtol=1e-4):
        maximum = 0.0
        for activation in [None, *activations]:
            expected = (
                reference_readout(activation)
                if reference_readout is not None
                else self.adapter.answer_token_logprobs(
                    prompt_tokens=self.prompt,
                    answer_tokens=self.answer,
                    site=self.site,
                    activation=activation,
                )
                .double()
                .mean()
            )
            actual = self(activation)
            error = float((actual - expected).abs().max())
            maximum = max(maximum, error)
            if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
                raise UnsupportedCapability(
                    f"Frozen prefix differs from full scoring: max_abs_error={error:.6g}"
                )
        return {
            "max_abs_error": maximum,
            "atol": atol,
            "rtol": rtol,
            "controls": 1 + len(activations),
        }

    def __call__(self, activation):
        return self.token_logprobs(activation).double().mean()
