"""Native NNsight 0.8 generation-step API contract without GPU downloads."""
from __future__ import annotations

import pytest
import torch

from saebase.api.nnsight_local import NNsightLocal, TraceLocation


class FakeResult:
    def __init__(self, model):
        self.model = model

    def save(self):
        token = 2 if self.model.layer.output[0, 0].item() > 0 else 7
        return torch.tensor([[1, token, 3]], dtype=torch.long)


class NativeTrace:
    def __init__(self, model):
        self.result = FakeResult(model)
        self.iter = self
        self.selections = []

    def __getitem__(self, steps):
        self.selections.extend(steps)
        return steps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class NativeNNsightModel:
    def __init__(self):
        self.layer = type("Envoy", (), {})()
        self.layer.output = torch.tensor([[2.0, 1.0]])
        self.generations = []

    def trace(self, inputs):
        raise RuntimeError("This test exercises NNsight generate(), not trace()")

    def generate(self, inputs, *, max_new_tokens, do_sample):
        assert inputs == "Example prompt"
        assert max_new_tokens == 3
        assert do_sample is False
        self.layer.output = torch.tensor([[2.0, 1.0]])
        run = NativeTrace(self)
        self.generations.append(run)
        return run


def test_nnsight_native_generation_step_edit_is_executed():
    model = NativeNNsightModel()
    api = NNsightLocal(model)
    result = api.generate_edit(
        "Example prompt",
        TraceLocation("layer"),
        steps=[0, 2],
        operation="zero",
        max_new_tokens=3,
    )
    assert result["executed"] is True
    assert result["selected_steps"] == [0, 2]
    assert result["step_interpretation"] == "forward_occurrences_0_is_prefill"
    assert torch.equal(result["baseline_token_ids"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(result["edited_token_ids"], torch.tensor([[1, 7, 3]]))
    assert result["same_continuation"] is False
    assert len(model.generations) == 2
    assert model.generations[1].selections == [0, 2]


@pytest.mark.parametrize("selected", [[], [3], [-1], [1, 1], [2, 1]])
def test_nnsight_generation_rejects_invalid_steps(selected):
    api = NNsightLocal(NativeNNsightModel())
    with pytest.raises(ValueError, match="steps"):
        api.generate_edit(
            "Example prompt",
            TraceLocation("layer"),
            steps=selected,
            operation="scale",
            max_new_tokens=3,
            value=0.5,
        )


def test_nnsight_generation_does_not_fake_an_unsupported_backend():
    model = NativeNNsightModel()
    model.generate = None
    with pytest.raises(RuntimeError, match="cannot trace generation"):
        NNsightLocal(model).generate_edit(
            "Example prompt", TraceLocation("layer"),
            steps=[0], operation="zero",
        )
