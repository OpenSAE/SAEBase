"""Exact suffix replay must match the original hook forward, including early edits."""

from types import SimpleNamespace

import pytest
import torch

from models.hf_hooks_backend import HFHooksModel
from saebase.api import ActivationSite, ModelAdapter
from saebase.interventions import BatchEngine, Condition, ExperimentIdentity, ResultWriter
from saebase.interventions.clean_prefix import (
    CLEAN_KV_STATUS,
    edited_logits_clean_prefix,
)
from saebase.store import sha256_state_dict
from examples._synthetic_common import ResearchToySAE
from tests.test_performance_hf_batch import records


class CausalMix(torch.nn.Module):
    def forward(self, hidden):
        earlier = torch.zeros_like(hidden)
        earlier[:, 1:] = hidden[:, :-1]
        return hidden * 1.1 + earlier * 0.25


class TinyStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.embed = torch.nn.Embedding(32, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([CausalMix(), CausalMix()])
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(128).float().reshape(32, 4) / 100)
            self.lm_head.weight.normal_(0, 0.2)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def wrapper():
    return HFHooksModel(TinyStack().eval(), None, model_name="tiny-stack", device="cpu")


def test_identity_and_early_edit_match_full_hook():
    model = wrapper()
    ids = torch.tensor([[1, 2, 3, 4]])
    for layer in (0, 1):
        cached = model.cache_site_hidden(ids, layer)
        replayed = model.logits_from_site_hidden(ids, layer, cached, readout_offset=0)
        reference = model(ids)[0, -1]
        torch.testing.assert_close(replayed, reference)
        edited = cached.clone()
        edited[0, 0] = edited[0, 0] + 0.5
        actual = model.logits_from_site_hidden(ids, layer, edited, readout_offset=0)
        expected = model.run_with_patch(ids, layer, {3: edited[0, 0]}, readout_offset=0)
        torch.testing.assert_close(actual, expected)


def test_failed_suffix_restores_the_original_blocks():
    model = wrapper()
    ids = torch.tensor([[1, 2, 3]])
    before = model(ids).clone()
    original = model._model.model.layers[0].forward
    cached = model.cache_site_hidden(ids, 0)

    def boom(hidden):
        raise RuntimeError("out of memory during suffix")

    model._model.model.layers[1].forward = boom
    with pytest.raises(RuntimeError, match="out of memory"):
        model.logits_from_site_hidden(ids, 0, cached)
    assert model._model.model.layers[0].forward == original
    model._model.model.layers[1].forward = CausalMix.forward.__get__(
        model._model.model.layers[1], CausalMix
    )
    torch.testing.assert_close(model(ids), before)


def test_unsupported_stack_is_refused():
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=1)

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=input_ids.float().unsqueeze(-1))

    model = HFHooksModel(Bare().eval(), None, model_name="bare", device="cpu")
    assert model.supports_suffix_replay() is False
    with pytest.raises(ValueError, match="unsupported"):
        model.cache_site_hidden(torch.tensor([[1, 2]]), 0)


def test_engine_suffix_matches_hook_and_is_opt_in(tmp_path):
    model = wrapper()
    adapter = ModelAdapter(model)
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    ident = ExperimentIdentity(
        sha256_state_dict(model),
        sha256_state_dict(sae),
        site,
        model_key="tiny-stack",
        sae_key="tiny",
        backend=adapter.backend,
    )
    samples = records()
    conditions = [Condition("identity", "identity"), Condition("ablate", "ablate", (0,))]
    reference = BatchEngine(adapter, sae, ident)
    compare = BatchEngine(adapter, sae, ident, enable_suffix_replay=True)
    for sample in samples:
        ids = torch.tensor([sample.tokens])
        site = ActivationSite(0, "blocks.0.hook_resid_post", sample.patch_position)
        original = adapter.capture(tokens=ids, site=site)
        edited = original + 0.3
        jobs = [(sample, original), (sample, edited)]
        torch.testing.assert_close(
            torch.stack(compare._suffix_patched_logits(jobs)),
            torch.stack(reference._patched_logits(jobs)),
        )
    with ResultWriter(tmp_path / "hook", ident) as writer:
        reference.run(samples, conditions, writer, sample_batch=3, microbatch=8)
    with ResultWriter(tmp_path / "suffix", ident) as writer:
        engine = BatchEngine(adapter, sae, ident, enable_suffix_replay=True)
        engine.run(samples, conditions, writer, sample_batch=3, microbatch=8)
        # The baseline caches each prompt once. Identity verification and both
        # conditions then replay only the blocks after the site.
        assert engine.counters["suffix_prefix_forwards"] == len(samples)
        assert engine.counters["suffix_replay_forwards"] == len(samples) + len(samples) * len(conditions)
    with pytest.raises(ValueError, match="cannot be combined"):
        BatchEngine(adapter, sae, ident, enable_suffix_replay=True, enable_hf_masked_batch=True)


def test_clean_prefix_edit_matches_full_patch_and_refuses_edited_kv():
    model = wrapper()
    adapter = ModelAdapter(model)
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    tokens = torch.tensor([[1, 2, 3, 4]])
    condition = Condition("scale", "scale", (0,), value=2.0)
    actual = edited_logits_clean_prefix(adapter, sae, site=site, tokens=tokens, condition=condition)
    from saebase.api import capture_features
    from saebase.interventions.operators import apply_condition

    snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
    patch = apply_condition(sae, snapshot, condition)
    expected = adapter.patch_logits(tokens=tokens, site=site, activation=patch.activation)
    torch.testing.assert_close(actual, expected)
    assert CLEAN_KV_STATUS == "BLOCKED"
