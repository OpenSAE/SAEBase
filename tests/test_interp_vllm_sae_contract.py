"""Contract tests against interp-engine's documented [tokens, width] capture API.

No vLLM dependency or GPU required; real worker correctness is a separate
mandatory RTX 5070 acceptance gate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from types import ModuleType

import pytest
import torch

from saebase.examples._synthetic_common import ResearchToySAE
from saebase.interventions.specs import Condition
from saebase.vllm_backend.sae_interventions import InterpVLLMSAE


@dataclass(frozen=True)
class Address:
    kind: str
    layer: int


class Engine:
    n_layers = 2
    hooks_available = True
    active = None

    def __init__(self, static=False):
        self.hooks_available = not static
        self.static_points = (Address("resid_post", 0),) if static else ()
        self.static_writes = (Address("resid_post", 0),) if static else ()


class Facade:
    def __init__(self, engine):
        self.engine = engine

    def capture(self, ids, points):
        h = torch.tensor(ids, dtype=torch.float32).unsqueeze(1).repeat(1, 4) / 10
        if self.engine.active is not None:
            spec, masked = self.engine.active
            delta = spec.layers[0].operations[0].vector
            for i in range(len(ids)):
                if i not in masked:
                    h[i] += torch.as_tensor(delta, dtype=h.dtype)
        return {point: h.clone() for point in points}

    def decode_residuals(self, residuals, *, detach=True):
        return residuals[:, :3].clone()

    def generate_text(self, ids, **kwargs):
        return "EDITED" if self.engine.active is not None else "CLEAN"

    def shutdown(self):
        return None


def setup_fake_interp(monkeypatch):
    root = ModuleType("interp_engine")
    root.__path__ = []
    root.Address = Address
    steer_module = ModuleType("interp_engine.steer")
    specs_module = ModuleType("interp_engine.steer_specs")
    sync_module = ModuleType("interp_engine.sync")

    @dataclass
    class AddSpec:
        vector: object
        scale: float

    @dataclass
    class LayerSteeringSpec:
        operations: list

    @dataclass
    class SteeringSpec:
        layers: dict
        point: str

    @contextmanager
    def steer(engine, spec, *, prompt_token_ids, position_mask, generated):
        assert not generated
        assert len(prompt_token_ids) >= 1
        assert engine.active is None
        engine.active = (spec, set(position_mask))
        try:
            yield
        finally:
            engine.active = None

    specs_module.AddSpec = AddSpec
    specs_module.LayerSteeringSpec = LayerSteeringSpec
    specs_module.SteeringSpec = SteeringSpec
    steer_module.steer = steer
    sync_module.sync_model = Facade
    for name, module in (
        ("interp_engine", root),
        ("interp_engine.steer", steer_module),
        ("interp_engine.steer_specs", specs_module),
        ("interp_engine.sync", sync_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("backend", ("vllm", "vllm-static"))
def test_real_capture_contract_and_scoped_feature_edit(monkeypatch, backend):
    setup_fake_interp(monkeypatch)
    model = Engine(static=backend == "vllm-static")
    adapter = InterpVLLMSAE(model, ResearchToySAE().eval(), layer=0, backend=backend)
    result = adapter.measure(
        [1, 2, 3],
        Condition("ablate", "ablate", (0,)),
        patch_position=1,
        next_token_probe=True,
        atol=1e-5,
        rtol=1e-4,
    )
    assert result.patch_position == 1
    torch.testing.assert_close(result.edited_residual, result.expected_residual)
    assert result.activation_error < 1e-5
    saved_capture = Facade.capture

    def leak(self, ids, points):
        tensors = saved_capture(self, ids, points)
        if self.engine.active is not None:
            _spec, masked = self.engine.active
            for point, tensor in list(tensors.items()):
                changed = tensor.clone()
                for index in masked:
                    changed[index, 0] = changed[index, 0] + 1
                tensors[point] = changed
        return tensors

    Facade.capture = leak
    try:
        with pytest.raises(RuntimeError, match="excluded prompt positions"):
            adapter.measure(
                [1, 2, 3],
                Condition("ablate", "ablate", (0,)),
                patch_position=1,
                next_token_probe=False,
            )
    finally:
        Facade.capture = saved_capture
    assert result.clean_next_text == "CLEAN"
    assert result.edited_next_text == "EDITED"
    assert model.active is None


def test_generation_only_and_undeclared_static_refused(monkeypatch):
    setup_fake_interp(monkeypatch)
    with pytest.raises(ValueError, match="no SAE activation hooks"):
        InterpVLLMSAE(Engine(), ResearchToySAE(), layer=0, backend="vllm-generate")
    bad = Engine(static=True)
    bad.static_writes = ()
    with pytest.raises(RuntimeError, match="both read and write taps"):
        InterpVLLMSAE(bad, ResearchToySAE(), layer=0, backend="vllm-static")
