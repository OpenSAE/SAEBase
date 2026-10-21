"""CPU tests for real tokenwise SAE edits and fixed feature-direction KV steering."""
from __future__ import annotations

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from saebase.interventions.hf_cached_steering import generate_live_sae_cached
from saebase.interventions.specs import Condition
from saebase.vllm_backend.fast_steering import (
    decoder_feature_direction,
    generate_decoder_direction_cached,
)
from tests.test_clean_kv_exact import build


def test_live_sae_kv_uses_one_forward_per_token_and_cleans_hook():
    model, _adapter, sae = build()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    raw = model._model
    original = raw(input_ids=ids, use_cache=True).logits[0, -1]
    before = raw.calls
    result = generate_live_sae_cached(
        model, sae, ids, layer=0, condition=Condition("id", "identity"),
        max_new_tokens=4, record_steps=True,
    )
    assert result.model_forwards == 4
    assert raw.calls - before == 4
    assert len(result.generated_token_ids) == 4
    assert len(result.steps) == 4
    assert result.generated_token_ids[0] == int(original.argmax().item())
    assert result.protocol == "persistent_live_sae_kv_v1"
    assert not model._layer_module(0)._forward_hooks


def test_live_sae_ablation_and_exception_restore_hook():
    model, _adapter, sae = build()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    result = generate_live_sae_cached(
        model, sae, ids, layer=0, condition=Condition("a", "ablate", (0,)),
        max_new_tokens=3, record_steps=True,
    )
    assert result.model_forwards == 3
    assert all(x.edited_feature_ids for x in result.steps)
    assert all(x.edit_l2 >= 0 for x in result.steps)
    with pytest.raises(IndexError):
        generate_live_sae_cached(
            model, sae, ids, layer=0,
            condition=Condition("bad", "ablate", (100000,)),
            max_new_tokens=1,
        )
    assert not model._layer_module(0)._forward_hooks


def test_live_sae_validates_budget_and_model():
    model, _adapter, sae = build()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="context"):
        generate_live_sae_cached(
            model, sae, ids, layer=0, condition=Condition("i", "identity"),
            max_new_tokens=4, max_context_tokens=5,
        )
    model._model.config.model_type = "other"
    with pytest.raises(ValueError, match="Qwen3"):
        generate_live_sae_cached(
            model, sae, ids, layer=0, condition=Condition("i", "identity"),
        )


class FakeInterp:
    n_layers = 4
    hooks_available = True
    graph_replay = False

    def __init__(self):
        self.active = False
        self.calls = []

    def refuses(self, point):
        return None


def install_interp(monkeypatch):
    root = ModuleType("interp_engine")
    root.__path__ = []

    class Address:
        def __init__(self, name, layer):
            self.name, self.layer = name, layer

    root.Address = Address
    steering = ModuleType("interp_engine.steer")
    specs = ModuleType("interp_engine.steer_specs")

    class AddSpec:
        def __init__(self, vector, scale, normalize=False):
            self.vector, self.scale, self.normalize = vector, scale, normalize

    class LayerSteeringSpec:
        def __init__(self, operations):
            self.operations = operations

    class SteeringSpec:
        def __init__(self, layers, point):
            self.layers, self.point = layers, point

    @contextmanager
    def steer(model, spec, *, prompt_token_ids, position_mask, generated):
        assert generated
        assert position_mask == list(range(len(prompt_token_ids) - 1))
        assert not model.active
        model.active = True
        model.calls.append(("steer", spec.layers[2].operations[0].scale))
        try:
            yield
        finally:
            model.active = False

    def generate_stream(model, tokens, *, max_tokens, temperature, seed, stop_at_eos):
        model.calls.append(("generate", model.active))
        for i in range(max_tokens):
            yield SimpleNamespace(token_id=10 + i if model.active else 20 + i)

    steering.steer = steer
    steering.generate_stream = generate_stream
    specs.AddSpec = AddSpec
    specs.LayerSteeringSpec = LayerSteeringSpec
    specs.SteeringSpec = SteeringSpec
    for name, module in (
        ("interp_engine", root), ("interp_engine.steer", steering),
        ("interp_engine.steer_specs", specs),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_fixed_decoder_direction_and_true_generated_steering(monkeypatch):
    install_interp(monkeypatch)
    _model, _adapter, sae = build()
    direction = decoder_feature_direction(sae, 0)
    assert direction.ndim == 1 and direction.numel() == sae.cfg.d_in
    model = FakeInterp()
    result = generate_decoder_direction_cached(
        model, sae, [1, 2, 3], layer=2, feature_id=0,
        strength=1.5, max_new_tokens=4, compare_unsteered=True,
    )
    assert result.generated_token_ids == (10, 11, 12, 13)
    assert result.baseline_token_ids == (20, 21, 22, 23)
    assert result.protocol == "persistent_decoder_direction_kv_v1"
    assert model.calls == [
        ("generate", False), ("steer", 1.5), ("generate", True)
    ]
    assert not model.active


def test_generation_only_backend_refused(monkeypatch):
    install_interp(monkeypatch)
    _model, _adapter, sae = build()
    candidate = FakeInterp()
    candidate.hooks_available = False
    with pytest.raises(RuntimeError, match="generation-only"):
        generate_decoder_direction_cached(
            candidate, sae, [1], layer=2, feature_id=0, strength=1.0
        )
