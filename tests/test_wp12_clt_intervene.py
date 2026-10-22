"""CPU numerical check: a CLT feature actually edits downstream outputs."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from saebase.sae.scope2_clt import CLTIdentity, Scope2CLT
from saebase.store import sha256_state_dict
from saebase.wp12.circuits import GraphEdge, GraphNode, GraphRecord
from saebase.wp12.clt_intervene import execute_clt_feature


class TinyCLTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(3):
            block = nn.Module()
            block.pre = nn.Identity()
            block.post = nn.Identity()
            self.model.layers.append(block)
        self.readout = nn.Linear(2, 3, bias=False)
        self.readout.weight.data.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )

    def forward(self, inputs):
        x = inputs
        for layer in self.model.layers:
            x = layer.pre(x)
            x = layer.post(x)
        return self.readout(x)


def test_clt_edit_changes_later_layers_only():
    weights = []
    for source in range(3):
        dec = torch.zeros(2, 3, 2)
        if source == 1:
            dec[0, 1, 0] = 1
            dec[0, 2, 0] = 1
        weights.append({
            "w_enc": torch.eye(2),
            "w_dec": dec,
            "b_enc": torch.zeros(2),
            "b_dec": torch.zeros(2),
            "threshold": torch.zeros(2),
        })
    clt = Scope2CLT(
        weights,
        config={
            "architecture": "jump_relu",
            "type": "clt",
            "hf_hook_point_in": "model.layers.{all}.pre.output",
            "hf_hook_point_out": "model.layers.{all}.post.output",
            "width": 6,
            "affine_connection": False,
        },
    )
    identity = CLTIdentity(
        repo_id="google/gemma-scope-2-270m-pt",
        revision="a" * 40,
        asset_directory="clt/width_6_l0_small",
        input_hook_template="model.layers.{all}.pre.output",
        output_hook_template="model.layers.{all}.post.output",
        model_name="synthetic-test-only",
        source_sha256="b" * 64,
        n_layers=3,
        d_model=2,
        features_per_layer=2,
        affine_skip=False,
    )
    model = TinyCLTModel()
    record = GraphRecord(
        nodes=(
            GraphNode("source-feature", "feature", 1, 0, 1, "source layer 1 F0"),
            GraphNode("native-error", "error", 1, None, 1, "unexplained"),
            GraphNode("output-logit", "logit", None, None, 1, "target logit"),
        ),
        edges=(
            GraphEdge("source-feature", "output-logit", 0.8),
            GraphEdge("native-error", "output-logit", -0.1),
        ),
        source_sha256="c" * 64,
        model_sha256=sha256_state_dict(model),
        transcoder_sha256=identity.source_sha256,
        original_tool_commit="d" * 40,
        source_format="cpu-numerical-fixture",
    )
    inputs = torch.tensor([[[0.5, 0.1], [2.0, 3.0], [1.0, 1.0]]])
    before = model(inputs).clone()
    result = execute_clt_feature(
        model, inputs, record, "source-feature", clt, identity, new_code=0.0,
    )
    assert result["executed"] is True
    assert result["target_layers"] == [1, 2]
    assert result["fired"] == {1: 1, 2: 1}
    assert result["activation_delta_l2"] > 0
    assert result["logit_delta_l2"] > 0
    assert torch.equal(before, model(inputs))
    assert torch.allclose(
        result["edited_logits"] - result["baseline_logits"],
        torch.tensor([-4.0, 0.0, -4.0]),
    )
    with pytest.raises(ValueError, match="checkpoint digest"):
        execute_clt_feature(
            model, inputs, record, "source-feature", clt,
            CLTIdentity(**{**identity.__dict__, "source_sha256": "f" * 64}),
        )
    unhooked = TinyCLTModel()
    unhooked.named_modules = lambda: iter(())
    with pytest.raises(ValueError, match="hook mapping"):
        execute_clt_feature(
            unhooked, inputs, record, "source-feature", clt, identity,
        )


def test_clt_rejects_graph_nonfeature_target():
    model = TinyCLTModel()
    clt = object()
    with pytest.raises(TypeError, match="typed CLT"):
        execute_clt_feature(model, torch.zeros(1, 2, 2), object(), "node", clt, object())
