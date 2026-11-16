"""Exact shared-prefix reuse must match independent reference computations."""

from types import SimpleNamespace

import torch

from tasks.intervene.backend.observations import PromptObservations, prompt_observation_session
from tasks.intervene.backend.steering import compare_steered_generation, GenerationSpec
from tests.product.test_feature_interventions import context, records, controls


def test_shared_observations_preserve_all_generation_steps_and_rng():
    model, sae, identity = context()
    pair = SimpleNamespace(model=model, sae=sae, site=identity.site)
    for temperature in [0.0, 0.7]:
        spec = GenerationSpec(max_new_tokens=5, seed=42, temperature=temperature)
        expected = [
            compare_steered_generation(
                model, sae, site=identity.site, sample=records()[0], condition=c, spec=spec
            )
            for c in controls()
        ]
        with prompt_observation_session(pair, True):
            actual = [
                compare_steered_generation(
                    model, sae, site=identity.site, sample=records()[0], condition=c, spec=spec
                )
                for c in controls()
            ]
        assert actual == expected


def test_clean_observations_reused_and_patch_bytes_bound_the_cache(monkeypatch):
    model, sae, identity = context()
    pair = SimpleNamespace(model=model, sae=sae, site=identity.site)
    tokens = torch.tensor([records()[0].tokens])
    calls = []
    original = model.forward_logits

    def counted(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(model, "forward_logits", counted)
    cache = PromptObservations(pair)
    first = cache.observe(tokens, identity.site)
    second = cache.observe(tokens.clone(), identity.site)
    assert first is second and len(calls) == 1
    replay = cache.patch(first[0], first[0].activation)
    changed = cache.patch(first[0], first[0].activation + 1)
    assert not torch.equal(replay, changed)
    assert torch.equal(replay, cache.patch(first[0], first[0].activation))


def test_native_context_limit_records_partial_generation_without_rewriting_input():
    model, sae, identity = context()
    sample = records()[0]
    model.model.cfg.n_ctx = len(sample.tokens)
    result = compare_steered_generation(
        model,
        sae,
        site=identity.site,
        sample=sample,
        condition=controls()[0],
        spec=GenerationSpec(max_new_tokens=5, stop_at_context_limit=True),
    )
    assert len(result["baseline_generated_token_ids"]) == 1
    assert len(result["steered_generated_token_ids"]) == 1
    assert result["baseline_stop_reason"] == result["steered_stop_reason"] == "context_limit"
    assert result["shared_prefix_tokens"] == len(sample.tokens)
