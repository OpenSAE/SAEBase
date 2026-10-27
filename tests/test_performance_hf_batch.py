"""Offline parity and batching regression tests for the optional HF hook path."""

from types import SimpleNamespace

import pytest
import torch

from models.hf_hooks_backend import HFHooksModel
from saebase.api import ModelAdapter, ActivationSite
from saebase.interventions import (
    BatchEngine,
    Condition,
    ExperimentIdentity,
    MeasurementSample,
    ResultWriter,
)
from saebase.store import SampleRecord, TokenSelection, sha256_state_dict
from saebase.saebase.examples._synthetic_common import ResearchToySAE


class Layer(torch.nn.Module):
    def forward(self, h):
        return h * 1.25 + 0.25


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.embed = torch.nn.Embedding(32, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.calls = 0
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(128).float().reshape(32, 4) / 100)
            self.lm_head.weight.fill_(0.1)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.calls += 1
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


def context():
    wrapper = HFHooksModel(TinyCausalLM().eval(), None, model_name="tiny", device="cpu")
    adapter = ModelAdapter(wrapper)
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    ident = ExperimentIdentity(
        sha256_state_dict(wrapper),
        sha256_state_dict(sae),
        site,
        model_key="tiny",
        sae_key="tiny",
        backend=adapter.backend,
    )
    return adapter, sae, ident


def records():
    return [
        MeasurementSample(
            SampleRecord(
                f"sample-{i}",
                ids,
                selection=TokenSelection("last_valid"),
                prompt_group_id=f"group-{i}",
                split="test",
            ),
            target_token_id=1,
        )
        for i, ids in enumerate(((1, 2, 3), (4, 5), (6, 7, 8, 9)))
    ]


def test_hf_batch_forward_matches_individual():
    adapter, _, _ = context()
    wrapper = adapter.model
    samples = records()
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    activations, logits = wrapper.run_batch(
        ids, layer=0, positions=[2, 1, 3], readout_positions=[2, 1, 3], attention_mask=mask
    )
    for i, sample in enumerate(samples):
        t = torch.tensor([sample.tokens])
        expected_h = adapter.capture(
            tokens=t, site=ActivationSite(0, "blocks.0.hook_resid_post", sample.patch_position)
        )
        expected_logits = adapter.forward_logits(tokens=t, readout_position=sample.readout_index)
        torch.testing.assert_close(activations[i], expected_h)
        torch.testing.assert_close(logits[i], expected_logits)
    replacements = activations + 0.2
    _, patched = wrapper.run_batch(
        ids,
        layer=0,
        positions=[2, 1, 3],
        readout_positions=[2, 1, 3],
        attention_mask=mask,
        patches=replacements,
    )
    for i, sample in enumerate(samples):
        t = torch.tensor([sample.tokens])
        expected = adapter.patch_logits(
            tokens=t,
            site=ActivationSite(0, "blocks.0.hook_resid_post", sample.patch_position),
            activation=replacements[i],
            readout_position=sample.readout_index,
        )
        torch.testing.assert_close(patched[i], expected)
    with pytest.raises(ValueError, match="right padding"):
        wrapper.run_batch(
            ids, layer=0, positions=[2, 1, 3], readout_positions=[2, 1, 3], attention_mask=1 - mask
        )


def test_identity_replay_oom_keeps_reduced_microbatch(tmp_path):
    adapter, sae, ident = context()
    samples = records()
    conditions = [Condition("identity", "identity"), Condition("ablate", "ablate", (0,))]
    engine = BatchEngine(adapter, sae, ident, pad_token_id=0)
    widths: list[int] = []
    original = engine._patched_logits
    state = {"large": False}

    def patched(jobs):
        widths.append(len(jobs))
        if len(jobs) > 2 and not state["large"]:
            state["large"] = True
            raise torch.cuda.OutOfMemoryError("injected out of memory")
        return original(jobs)

    engine._patched_logits = patched
    with ResultWriter(tmp_path, ident) as writer:
        stats = engine.run(samples, conditions, writer, sample_batch=3, microbatch=8)
    assert stats.measured == len(samples) * len(conditions)
    assert stats.microbatch_retries >= 1
    assert widths[0] > 2
    assert all(width <= 2 for width in widths[1:])


def test_baseline_oom_splits_without_dropping_samples(tmp_path):
    adapter, sae, ident = context()
    samples = records()
    conditions = [Condition("identity", "identity")]
    engine = BatchEngine(adapter, sae, ident, pad_token_id=0)
    original = engine._baselines
    state = {"full": False}

    def baselines(active):
        if len(active) > 1 and not state["full"]:
            state["full"] = True
            raise torch.cuda.OutOfMemoryError("injected out of memory")
        return original(active)

    engine._baselines = baselines
    with ResultWriter(tmp_path, ident) as writer:
        stats = engine.run(samples, conditions, writer, sample_batch=3, microbatch=4)
    assert stats.measured == len(samples)
    assert stats.microbatch_retries >= 1


def test_hf_masked_batch_defaults_to_single_prompt(tmp_path):
    adapter, sae, ident = context()
    samples = records()
    conditions = [Condition("identity", "identity")]
    with ResultWriter(tmp_path, ident) as writer:
        stats = BatchEngine(adapter, sae, ident, pad_token_id=0).run(
            samples, conditions, writer, sample_batch=3, microbatch=4
        )
    assert stats.measured == len(samples)
    assert stats.baseline_batches == len(samples)


def test_batch_engine_hf_uses_one_forward_per_chunk(tmp_path):
    adapter, sae, ident = context()
    samples = records()
    conditions = [Condition("identity", "identity"), Condition("ablate", "ablate", (0,))]
    with ResultWriter(tmp_path, ident) as writer:
        stats = BatchEngine(adapter, sae, ident, pad_token_id=0, enable_hf_masked_batch=True).run(
            samples, conditions, writer, sample_batch=3, microbatch=4
        )
    assert stats.measured == len(samples) * len(conditions)
    assert stats.baseline_batches == 1
    assert stats.patch_batches < stats.measured + len(samples)
