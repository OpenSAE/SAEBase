"""One HF forward returns the site activation and the baseline logits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.hf_hooks_backend import HFHooksModel
from saebase.api import ActivationSite, ModelAdapter
from saebase.interventions import (
    BatchEngine,
    Condition,
    ExperimentIdentity,
    MeasurementSample,
    ResultWriter,
)
from saebase.store import SampleRecord, TokenSelection, sha256_state_dict


class Block(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1


class Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block()])


class CountingLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Inner()
        self.calls = 0
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.marker = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor | None = None) -> SimpleNamespace:
        self.calls += 1
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4).contiguous()
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = hidden.sum(dim=-1, keepdim=True).expand(-1, -1, 5).contiguous()
        return SimpleNamespace(logits=logits)


def _hooks() -> tuple[HFHooksModel, CountingLM]:
    raw = CountingLM()
    model = HFHooksModel(raw, SimpleNamespace(), model_name="tiny", device="cpu")
    return model, raw


def test_one_forward_matches_separate_capture_and_logits():
    model, raw = _hooks()
    ids = torch.tensor([[1, 2, 4]])
    activation = model.get_resid(ids, 0, [0])
    logits = model(ids)[0, -1]
    assert raw.calls == 2
    combined_h, combined_y = model.capture_with_logits(ids, 0, [0], readout_offset=0)
    assert raw.calls == 3
    torch.testing.assert_close(combined_h, activation)
    torch.testing.assert_close(combined_y, logits)


def test_early_readout_and_hook_cleanup_on_failure():
    model, raw = _hooks()
    ids = torch.tensor([[1, 2, 4, 7]])
    activation = model.get_resid(ids, 0, [2])
    logits = model(ids)[0, 1]
    combined_h, combined_y = model.capture_with_logits(ids, 0, [2], readout_offset=2)
    torch.testing.assert_close(combined_h, activation)
    torch.testing.assert_close(combined_y, logits)

    def boom(input_ids: torch.Tensor | None = None) -> SimpleNamespace:
        raise RuntimeError("boom")

    raw.forward = boom
    layer = model._layer_module(0)
    with pytest.raises(RuntimeError, match="boom"):
        model.capture_with_logits(ids, 0, [0])
    assert len(layer._forward_hooks) == 0


def test_engine_uses_one_forward_per_uncached_hf_sample(tmp_path):
    model, raw = _hooks()
    adapter = ModelAdapter(model)
    sae_weight = nn.Parameter(torch.ones(4))

    class TinySAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = sae_weight
            self.cfg = SimpleNamespace(d_in=4, d_sae=4, normalize_activations=None)

        def encode(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden

        def decode(self, codes: torch.Tensor) -> torch.Tensor:
            return codes

    sae = TinySAE()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    identity = ExperimentIdentity(
        sha256_state_dict(model),
        sha256_state_dict(sae),
        site,
        model_key="tiny",
        sae_key="tiny",
        backend=adapter.backend,
    )
    sample = MeasurementSample(
        SampleRecord(
            "s", (1, 2, 4), selection=TokenSelection("last_valid"), prompt_group_id="g", split="t"
        ),
        target_token_id=0,
    )
    before = raw.calls
    with ResultWriter(tmp_path, identity) as writer:
        BatchEngine(adapter, sae, identity, pad_token_id=0).run(
            [sample],
            [Condition("identity", "identity")],
            writer,
            verify_identity=False,
            tolerance=1e-5,
        )
    assert raw.calls - before == 2
