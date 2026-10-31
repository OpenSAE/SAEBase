"""No-download numerical tests for real Gemma Scope 2 JumpReLU algebra."""

from __future__ import annotations

import pytest
import torch

from saebase.sae.scope2_native import (
    JumpReLURepresentation,
    _validate_source,
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
        "w_enc": w_enc,
        "w_dec": w_dec,
        "b_enc": torch.zeros(4),
        "b_dec": torch.zeros(2, 3),
        "threshold": torch.zeros(4),
    }
    model = JumpReLURepresentation(
        params, config={"architecture": "jump_relu", "width": 4, "affine_connection": False}
    )
    x = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
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
    assert (
        _validate_source(
            "google/gemma-scope-2-270m-pt",
            "a" * 40,
            "transcoder/layer_12_width_65k_l0_medium",
        )
        == "transcoder"
    )
    assert (
        _validate_source(
            "google/gemma-scope-2-270m-pt",
            "a" * 40,
            "crosscoder/layer_5_9_12_15_width_65k_l0_medium",
        )
        == "crosscoder"
    )
    with pytest.raises(ValueError):
        _validate_source("random/host", "a" * 40, "transcoder/foo")
    with pytest.raises(ValueError):
        _validate_source("google/gemma-scope-2-270m-pt", "main", "transcoder/foo")
    with pytest.raises(ValueError):
        _validate_source("google/gemma-scope-2-270m-pt", "a" * 40, "../../private")


def test_p6_native_transcoder_circuit_intervention_changes_real_logits():
    from torch import nn

    from saebase.sae.scope2_native import ScopeIdentity
    from saebase.store import sha256_state_dict
    from saebase.circuits.transcoder_intervention import execute_transcoder_node
    from saebase.circuits.graph import GraphNode, GraphEdge, GraphRecord

    class TinyForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_layer = nn.Linear(2, 2, bias=False)
            self.out_layer = nn.Linear(2, 2, bias=False)
            self.readout = nn.Linear(2, 3, bias=False)
            for layer in (self.in_layer, self.out_layer):
                layer.weight.data.copy_(torch.eye(2))
            self.readout.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

        def forward(self, batch):
            return self.readout(self.out_layer(self.in_layer(batch)))

    model = TinyForward()
    params = {
        "w_enc": torch.eye(2),
        "w_dec": torch.eye(2),
        "b_enc": torch.zeros(2),
        "b_dec": torch.zeros(2),
        "threshold": torch.zeros(2),
    }
    native = JumpReLURepresentation(
        params, config={"width": 2, "architecture": "jump_relu", "affine_connection": False}
    )
    source_sha = "b" * 64
    info = ScopeIdentity(
        repo_id="google/gemma-scope-2-270m-pt",
        revision="a" * 40,
        asset_directory="transcoder/layer_0_width_2_l0_medium",
        family="transcoder",
        kind="transcoder",
        input_hook="in_layer.output",
        output_hook="out_layer.output",
        source_sha256=source_sha,
        width=2,
        input_dim=2,
        output_dim=2,
        affine_skip=False,
        source_model="synthetic-cpu-only",
    )
    graph = GraphRecord(
        nodes=(
            GraphNode("f0", "feature", 0, 0, 1, "feature 0"),
            GraphNode("err", "error", 0, None, 1, "original error"),
            GraphNode("logit", "logit", None, None, 1, "target"),
        ),
        edges=(
            GraphEdge("f0", "logit", 0.7),
            GraphEdge("err", "logit", -0.1),
        ),
        source_sha256="c" * 64,
        model_sha256=sha256_state_dict(model),
        transcoder_sha256=source_sha,
        original_tool_commit="d" * 40,
        source_format="synthetic-numerical-fixture",
    )
    prompt = torch.tensor([[[0.5, 0.0], [2.0, 3.0], [1.0, 1.0]]])
    baseline = model(prompt).detach().clone()
    outcome = execute_transcoder_node(model, prompt, graph, "f0", native, info, new_code=0)
    assert outcome["executed"] is True
    assert outcome["activation_delta_l2"] > 0
    assert outcome["logit_delta_l2"] > 0
    assert outcome["fired"] == {"input": 1, "output": 1}
    assert torch.equal(model(prompt), baseline)
    assert torch.allclose(
        outcome["edited_logits"] - outcome["baseline_logits"], torch.tensor([-2.0, 0.0, -2.0])
    )
    with pytest.raises(ValueError, match="checkpoint disagree"):
        execute_transcoder_node(
            model,
            prompt,
            graph,
            "f0",
            native,
            ScopeIdentity(**{**info.__dict__, "source_sha256": "0" * 64}),
        )


def test_scope2_pinned_native_loader_reads_real_source_contract(monkeypatch, tmp_path):
    import json
    import sys
    from types import ModuleType

    from saebase.sae.scope2_native import load_official_scope2

    cfg = {
        "hf_hook_point_in": "model.layers.12.pre_feedforward_layernorm.output",
        "hf_hook_point_out": "model.layers.12.post_feedforward_layernorm.output",
        "width": 2,
        "model_name": "google/gemma-3-270m-pt",
        "architecture": "jump_relu",
        "l0": 30,
        "affine_connection": False,
        "type": "transcoder",
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    tensor_file = tmp_path / "params.safetensors"
    tensor_file.write_bytes(b"synthetic-only-test-stub")

    def download(*, repo_id, filename, revision):
        assert repo_id == "google/gemma-scope-2-270m-pt"
        assert revision == "a" * 40
        assert filename.startswith("transcoder/layer_12_width_2_l0_medium/")
        return str(config if filename.endswith("config.json") else tensor_file)

    def read_safe(_path, *, device):
        assert device == "cpu"
        return {
            "w_enc": torch.eye(2),
            "w_dec": torch.eye(2),
            "b_enc": torch.zeros(2),
            "b_dec": torch.zeros(2),
            "threshold": torch.zeros(2),
        }

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    safe = ModuleType("safetensors")
    safe.__path__ = []
    backend = ModuleType("safetensors.torch")
    backend.load_file = read_safe
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "safetensors", safe)
    monkeypatch.setitem(sys.modules, "safetensors.torch", backend)

    model, identity = load_official_scope2(
        repo_id="google/gemma-scope-2-270m-pt",
        revision="a" * 40,
        directory="transcoder/layer_12_width_2_l0_medium",
    )
    assert identity.kind == "transcoder"
    assert identity.input_hook != identity.output_hook
    assert identity.width == 2
    assert len(identity.source_sha256) == 64
    assert torch.equal(model(torch.ones(1, 2)), torch.ones(1, 2))


def test_scope2_manifest_indexes_published_transcoder_crosscoder_clt():
    from types import SimpleNamespace

    from saebase.sae.catalog import discover_official_scope_assets

    names = [
        "transcoder/layer_12_width_65k_l0_medium_affine/params.safetensors",
        "crosscoder/layer_5_9_12_15_width_65k_l0_medium/params.safetensors",
        "clt/width_262k_l0_medium_affine/params.safetensors",
    ]
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[SimpleNamespace(rfilename=name) for name in names],
    )
    assets = discover_official_scope_assets("google/gemma-scope-2-270m-pt", info=info)
    assert len(assets) == 3
    by_type = {asset.activation_family: asset for asset in assets}
    assert by_type["transcoder"].affine_skip
    assert by_type["transcoder"].layer == 12
    assert by_type["crosscoder"].layers == (5, 9, 12, 15)
    assert by_type["clt"].layer == -1
    assert by_type["clt"].affine_skip
