"""Scoped reuse of exact clean-prefix observations across independent edits.

No KV history is reused: only results for byte-identical complete token prefixes,
SAE sites and patched activations are memoized within one read-only pair/prompt.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from saebase.api import capture_features, edit_features, InterventionResult

_ACTIVE = ContextVar("saebase_prompt_observations", default=None)


class PromptObservations:
    def __init__(self, pair, max_entries=64):
        self.pair = pair
        self.max_entries = max_entries
        self.clean = OrderedDict()
        self.edited = OrderedDict()
        self.verified = set()

    def _key(self, tokens, site):
        return (
            tuple(tokens[0].detach().cpu().tolist()),
            site.layer,
            site.hook_name,
            site.position % int(tokens.shape[1]),
        )

    def observe(self, tokens, site):
        key = self._key(tokens, site)
        if key not in self.clean:
            snapshot = capture_features(self.pair.model, self.pair.sae, site=site, tokens=tokens)
            logits = self.pair.model.forward_logits(tokens=tokens, readout_position=-1)
            self.clean[key] = (snapshot, logits)
            if len(self.clean) > self.max_entries:
                self.clean.popitem(last=False)
        self.clean.move_to_end(key)
        return self.clean[key]

    def patch(self, snapshot, activation):
        key = (
            self._key(snapshot.tokens, snapshot.site),
            str(activation.dtype),
            activation.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes(),
        )
        if key not in self.edited:
            self.edited[key] = self.pair.model.patch_logits(
                tokens=snapshot.tokens,
                site=snapshot.site,
                activation=activation,
                readout_position=-1,
            )
            if len(self.edited) > self.max_entries:
                self.edited.popitem(last=False)
        self.edited.move_to_end(key)
        return self.edited[key]

    def evaluate(self, prompt, replacements, verify_identity):
        tokens = self.pair.model.tokens(prompt=prompt)
        snapshot, baseline = self.observe(tokens, self.pair.site)
        key = self._key(tokens, self.pair.site)
        if verify_identity and key not in self.verified:
            replay = self.patch(snapshot, snapshot.activation)
            if not torch.allclose(replay.float(), baseline.float(), rtol=1e-4, atol=1e-4):
                raise RuntimeError("identity patch changes model output during shared observation")
            self.verified.add(key)
        patch = edit_features(self.pair.sae, snapshot, replacements=replacements)
        edited = self.patch(snapshot, patch.activation)
        if edited.shape != baseline.shape or not bool(torch.isfinite(edited).all()):
            raise RuntimeError("edited logits must be finite and match baseline shape")
        return InterventionResult(snapshot, patch, baseline, edited)


def current_observations(model, sae):
    active = _ACTIVE.get()
    return (
        active
        if active is not None and active.pair.model is model and active.pair.sae is sae
        else None
    )


@contextmanager
def prompt_observation_session(pair, enabled):
    if not enabled:
        yield
        return
    token = _ACTIVE.set(PromptObservations(pair))
    try:
        yield
    finally:
        _ACTIVE.reset(token)
