"""No-download numerical tests for real Gemma Scope 2 JumpReLU algebra."""
from __future__ import annotations

import pytest
import torch

from saebase.sae.scope2_native import (
    JumpReLURepresentation, _validate_source,
)


def test_single_layer_jumprelu_codes_and_decoder():
    params = {
        "w_enc": torch.eye(3),
        "w_dec": torch.eye(3),
        "b_enc": torch.zeros(3),
        "b_dec": torch.zeros(3),
        "threshold": torch.tensor([0.2, 0.5, 0.1]),
    }
    model = JumpReLURepresentation(
        params, config={"architecture": "jump_relu", "width": 3, "affine_connection": False}
    )
    x = torch.tensor([[0.4, 0.3, -1.0]])
    assert torch.equal(model.encode(x), torch.tensor([[0.4, 0.0, 0.0]]))
    assert torch.equal(model(x), torch.tensor([[0.4, 0.0, 0.0]]))


def test_affine_transcoder_is_distinct_from_decoder_code_edit():
    params = {
        "w_enc": torch.eye(2),
        "w_dec": torch.eye(2),
        "b_enc": torch.zeros(2),
        "b_dec": torch.zeros(2),
        "threshold": torch.zeros(2),
        "affine_skip_connection": torch.eye(2),
    }
    model = JumpReLURepresentation(
        params, config={"architecture": "jump_relu", "width": 2, "affine_connection": True}
    )
    x = torch.tensor([[3.0, 2.0]])
    assert torch.equal(model.encode(x), x)
    assert torch.equal(model.decode(model.encode(x)), x)
    assert torch.equal(model(x), 2 * x)
    with pytest.raises(ValueError, match="skip matrix"):
        JumpReLURepresentation(
            {k: v for k, v in params.items() if k != "affine_skip_connection"},
            config={"architecture": "jump_relu", "width": 2, "affine_connection": True},
        )


def test_partial_crosscoder_keeps_all_layer_axes_and_errors():
    w_enc = torch.zeros(2, 3, 4)
    w_enc[0, 0, 0] = 1
    w_enc[1, 1, 1] = 1
    w_dec = torch.zeros(4, 2, 3)
    w_dec[0, 0, 0] = 1
    w_dec[1, 1, 1] = 1
    params = {
        "w_enc": w_enc, "w_dec": w_dec,
        "b_enc": torch.zeros(4), "b_dec": torch.zeros(2, 3),
        "threshold": torch.zeros(4),
    }
    model = JumpReLURepresentation(
        params, config={"architecture": "jump_relu", "width": 4, "affine_connection": False}
    )
    x = torch.tensor([[[2., 0., 0.], [0., 4., 0.]]])
    assert model.encode(x).tolist() == [[2, 4, 0, 0]]
    assert torch.equal(model(x), x)
    with pytest.raises(ValueError, match="all declared input layers"):
        model.encode(x[:, :1, :])
    with pytest.raises(ValueError, match="orientation"):
        JumpReLURepresentation(
            {**params, "w_dec": torch.zeros(3, 2, 4)},
            config={"architecture": "jump_relu", "width": 4, "affine_connection": False},
        )


def test_only_official_pinned_sources_allowed():
    assert _validate_source(
        "google/gemma-scope-2-270m-pt", "a" * 40,
        "transcoder/layer_12_width_65k_l0_medium",
    ) == "transcoder"
    assert _validate_source(
        "google/gemma-scope-2-270m-pt", "a" * 40,
        "crosscoder/layer_5_9_12_15_width_65k_l0_medium",
    ) == "crosscoder"
    with pytest.raises(ValueError):
        _validate_source("random/host", "a" * 40, "transcoder/foo")
    with pytest.raises(ValueError):
        _validate_source("google/gemma-scope-2-270m-pt", "main", "transcoder/foo")
    with pytest.raises(ValueError):
        _validate_source("google/gemma-scope-2-270m-pt", "a" * 40, "../../private")
