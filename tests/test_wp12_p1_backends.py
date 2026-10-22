"""Offline P1 backend contracts; no upstream weights or GPU downloads."""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from models.transformerlens_backend import boot_transformerlens
from saebase.api import (
    ActivationSite, ModelAdapter, NNsightLocal, TraceLocation,
    capture_block_components,
)
from saebase.api.model import UnsupportedCapability
from saebase.sae.catalog import (
    discover_official_scope_assets, registered_scope_assets,
)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.compatibility = None

    def enable_compatibility_mode(self, *, no_processing=False):
        self.compatibility = no_processing


def test_legacy_loader_keeps_old_processing(monkeypatch):
    calls = []

    class Legacy:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("processed", name, kwargs))
            return SmallModel()

        @staticmethod
        def from_pretrained_no_processing(name, **kwargs):
            calls.append(("raw", name, kwargs))
            return SmallModel()

    monkeypatch.setattr(
        "models.hf_compat.import_transformer_lens",
        lambda: SimpleNamespace(HookedTransformer=Legacy),
    )
    model, engine = boot_transformerlens(
        "pythia-70m-deduped", device="cpu", dtype=torch.float32,
        no_processing=True,
    )
    assert engine == "hooked_transformer_legacy"
    assert calls[0][0] == "raw"
    assert model._saebase_weight_processing == "raw_hf"
    assert not model.weight.requires_grad


@pytest.mark.parametrize("raw", [True, False])
def test_tl4_bridge_uses_explict_legacy_alias_mode(monkeypatch, raw):
    calls = []

    class Bridge:
        @staticmethod
        def boot_transformers(name, **kwargs):
            calls.append((name, kwargs))
            return SmallModel()

    package = ModuleType("transformer_lens")
    package.__path__ = []
    submodule = ModuleType("transformer_lens.model_bridge")
    submodule.TransformerBridge = Bridge
    monkeypatch.setitem(sys.modules, "transformer_lens", package)
    monkeypatch.setitem(sys.modules, "transformer_lens.model_bridge", submodule)
    monkeypatch.setattr(
        "models.hf_compat.import_transformer_lens",
        lambda: SimpleNamespace(),
    )
    model, engine = boot_transformerlens(
        "gpt2", device="cpu", dtype=None, no_processing=raw,
    )
    assert calls == [("gpt2", {"device": "cpu", "dtype": torch.float32})]
    assert engine == "transformer_bridge_local"
    assert model.compatibility is raw
    assert model._saebase_weight_processing == (
        "raw_hf" if raw else "legacy_compatible"
    )


class HookToy(nn.Module):
    """A source-faithful tiny hook mock to check SAEBase's own contract."""

    def __init__(self, *, actually_fire=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.cfg = SimpleNamespace(n_layers=1)
        self.hook_dict = {
            f"blocks.0.hook_resid_{s}": object() for s in ("pre", "mid", "post")
        }
        self.actually_fire = actually_fire

    def get_hook_point(self, name):
        return self.hook_dict.get(name)

    def _activation(self, tokens, hook_name):
        base = tokens.float().unsqueeze(-1).expand(-1, -1, 4)
        increment = {"pre": 0.0, "mid": 1.0, "post": 3.0}[hook_name.rsplit("_", 1)[-1]]
        return base + increment

    def forward(self, tokens):
        h = self._activation(tokens, "blocks.0.hook_resid_post")
        return torch.stack([h.sum(-1), -h.sum(-1)], dim=-1)

    def run_with_cache(self, tokens, *, names_filter):
        return self(tokens), {
            names_filter: self._activation(tokens, names_filter)
        } if names_filter in self.hook_dict else {}

    def run_with_hooks(self, tokens, *, fwd_hooks):
        hook_name, edit = fwd_hooks[0]
        if self.actually_fire:
            original = self._activation(tokens, hook_name)
            patched = edit(original, hook="keyword_callback")
            return torch.stack([patched.sum(-1), -patched.sum(-1)], dim=-1)
        return self(tokens)


def test_hook_patch_requires_real_execution():
    adapter = ModelAdapter(HookToy())
    ids = torch.tensor([[1, 2, 3]])
    site = ActivationSite(0, "blocks.0.hook_resid_post", -1)
    activation = adapter.capture(tokens=ids, site=site)
    baseline = adapter.forward_logits(tokens=ids)
    patched = adapter.patch_logits(tokens=ids, site=site, activation=activation)
    assert torch.allclose(baseline, patched)
    assert len(adapter.available_hooks(layer=0)) == 3
    with pytest.raises(UnsupportedCapability, match="no fireable hook"):
        adapter.capture(tokens=ids, site=ActivationSite(0, "blocks.0.hook_wrong"))
    with pytest.raises(UnsupportedCapability, match="no fireable hook"):
        adapter.patch_logits(
            tokens=ids, site=ActivationSite(0, "blocks.0.hook_wrong"),
            activation=activation,
        )


def test_silent_patch_is_rejected():
    adapter = ModelAdapter(HookToy(actually_fire=False))
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(UnsupportedCapability, match="fired 0 times"):
        adapter.patch_logits(
            tokens=ids, site=ActivationSite(0, "blocks.0.hook_resid_post"),
            activation=torch.zeros(4),
        )


def test_block_components_are_real_differences():
    adapter = ModelAdapter(HookToy())
    components = capture_block_components(
        adapter, tokens=torch.tensor([[1, 2, 3]]), layer=0,
    )
    assert torch.allclose(components.attention_delta, torch.ones(4))
    assert torch.allclose(components.mlp_delta, torch.full((4,), 2.0))
    components.check()


@pytest.mark.parametrize("path", [
    "", "_hidden.layer", "model.__class__", "model.layers[-1]",
    "model.layers[x]", "model.layers[0].__dict__",
])
def test_nnsight_location_rejects_ambiguous_paths(path):
    with pytest.raises(ValueError):
        TraceLocation(path)


def test_nnsight_requires_a_real_trace_wrapper():
    with pytest.raises(TypeError):
        NNsightLocal(object())
    assert TraceLocation("model.layers[1].mlp", output_index=0).output_index == 0
    assert TraceLocation("[0]").path == "[0]"


def test_scope_catalog_lists_real_registered_profiles():
    assets = registered_scope_assets(model_key="gemma270m")
    assert assets and any(a.layer == 5 and a.width == "65k" for a in assets)
    assert all(a.status == "registered_not_live_verified" for a in assets)


def test_official_scope_manifest_is_not_a_fabricated_cartesian_product():
    repo = "google/gemma-scope-2-270m-pt"
    sibling = SimpleNamespace(rfilename="resid_post/layer_5_width_65k_l0_medium/params.npz")
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[sibling, sibling, SimpleNamespace(rfilename="LICENSE")],
    )
    assets = discover_official_scope_assets(repo, info=info)
    assert len(assets) == 1
    assert assets[0].layer == 5
    assert assets[0].width == "65k"
    assert assets[0].revision == "a" * 40
    assert assets[0].status == "official_manifest_unloaded"
    with pytest.raises(ValueError, match="unregistered"):
        discover_official_scope_assets("some/other-model", info=info)
