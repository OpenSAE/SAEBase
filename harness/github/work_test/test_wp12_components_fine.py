"""Numerical source-contract tests for attention-head and MLP-neuron contributions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from saebase.api.model import ModelAdapter, UnsupportedCapability
from saebase.api.fine_components import capture_projected_components


class ActualNativeModule(nn.Module):
    def __init__(self):
        super().__init__()
        attn = nn.Module()
        attn.W_O = nn.Parameter(
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                ]
            )
        )
        attn.b_O = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
        mlp = nn.Module()
        mlp.W_out = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
        )
        mlp.b_out = nn.Parameter(torch.tensor([0.3, 0.2, 0.1]))
        block = nn.Module()
        block.attn = attn
        block.mlp = mlp
        self.blocks = nn.ModuleList([block])
        self.cfg = SimpleNamespace(n_layers=1)

    def run_with_cache(self, tokens, *, names_filter):
        n = tokens.shape[1]
        z = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [0.0, 3.0]],
                [[4.0, 0.0], [2.0, 2.0]],
            ]
        )[:n].unsqueeze(0)
        post = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 2.0, 1.0, 4.0],
                [2.0, 0.0, 1.0, 3.0],
            ]
        )[:n].unsqueeze(0)
        attn_out = (
            torch.einsum("bthd,hdm->btm", z, self.blocks[0].attn.W_O) + self.blocks[0].attn.b_O
        )
        mlp_out = post @ self.blocks[0].mlp.W_out + self.blocks[0].mlp.b_out
        all_cache = {
            "blocks.0.attn.hook_z": z,
            "blocks.0.hook_attn_out": attn_out,
            "blocks.0.mlp.hook_post": post,
            "blocks.0.hook_mlp_out": mlp_out,
        }
        return torch.empty(1), {k: v for k, v in all_cache.items() if names_filter(k)}


@pytest.mark.parametrize(
    "family,count",
    [
        ("attention_heads", 2),
        ("mlp_neurons", 4),
    ],
)
def test_native_components_are_actual_vectors_and_reconstruct_measured_output(family, count):
    model = ModelAdapter(ActualNativeModule())
    value = capture_projected_components(
        model,
        tokens=torch.tensor([[1, 2, 3]]),
        layer=0,
        position=-1,
        component=family,
    )
    assert value.n_components == count
    assert value.vectors.shape == (count, 3)
    assert value.reconstruction_error_max < 1e-5
    assert torch.allclose(value.projected_sum(), value.measured_output)
    assert value.token_position == 2


def test_native_components_do_not_pass_wrong_weights():
    model = ActualNativeModule()

    def wrong(*args, **kwargs):
        res, cache = ActualNativeModule.run_with_cache(model, *args, **kwargs)
        cache["blocks.0.hook_attn_out"] = (
            cache.get("blocks.0.hook_attn_out", torch.zeros(1, 3, 3)) + 4
        )
        return res, cache

    model.run_with_cache = wrong
    with pytest.raises(RuntimeError, match="reconstitute"):
        capture_projected_components(
            ModelAdapter(model),
            tokens=torch.tensor([[1, 2, 3]]),
            layer=0,
            component="attention_heads",
        )


def test_unsupported_component_fails_instead_of_inventing_zero_vectors():
    model = ModelAdapter(ActualNativeModule())
    with pytest.raises(UnsupportedCapability):
        capture_projected_components(
            model,
            tokens=torch.tensor([[1]]),
            layer=1,
        )
    with pytest.raises(ValueError):
        capture_projected_components(
            model,
            tokens=torch.tensor([[1]]),
            layer=0,
            component="attention_logits",
        )
