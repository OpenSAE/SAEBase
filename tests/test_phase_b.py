"""Phase B: standalone API tests without model checkpoints or network access."""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from saebase.api import (
    ActivationSite, ModelAdapter, SAEIdentity,
    UnsupportedCapability, capture_features, coalition_replacements,
    edit_features, evaluate_edit,
)


class TinyModel(torch.nn.Module):
    """Simple causal residual model exposing TransformerLens-like API."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, 13, dtype=torch.float).reshape(3, 4) / 10)
        self.cfg = SimpleNamespace(n_layers=4, device="cpu")
        self.last_hook = "blocks.2.hook_resid_post"

    def to_tokens(self, text):
        return torch.tensor([[int(t) for t in text.split()]], dtype=torch.long)

    def _hidden(self, tokens):
        x = tokens.float().unsqueeze(-1)
        return torch.cat([x / 7, x / 11 + 1, x / 13 + 2], dim=-1)

    def _out(self, h):
        return torch.cumsum(h, dim=1) @ self.weight

    def forward(self, tokens):
        return self._out(self._hidden(tokens))

    def run_with_cache(self, tokens, names_filter=None):
        if names_filter != self.last_hook:
            return self(tokens), {}
        return self(tokens), {self.last_hook: self._hidden(tokens)}

    def run_with_hooks(self, tokens, fwd_hooks):
        h = self._hidden(tokens)
        for name, fn in fwd_hooks:
            if name != self.last_hook:
                raise ValueError("unknown hook")
            h = fn(h, None)
        return self._out(h)

    def generate(self, tokens, **kwargs):
        return torch.cat([tokens, torch.ones((1, 1), dtype=torch.long)], dim=1)


class TinyHFHooks(TinyModel):
    _saebase_backend = "hf_hooks"

    def get_resid(self, input_ids, layer, offsets):
        if layer != 2:
            raise ValueError("bad layer")
        h = self._hidden(input_ids)
        return torch.stack([h[0, h.shape[1] - 1 - off] for off in offsets])

    def run_with_patch(self, input_ids, layer, patches, *, readout_offset=0):
        if layer != 2:
            raise ValueError("bad layer")
        h = self._hidden(input_ids).clone()
        for off, act in patches.items():
            h[0, h.shape[1] - 1 - off] = act.to(h.dtype)
        return self._out(h)[0, h.shape[1] - 1 - readout_offset]


class AffineSAE(torch.nn.Module):
    def __init__(self, *, d_in=3, d_sae=5):
        super().__init__()
        self.cfg = SimpleNamespace(d_in=d_in, d_sae=d_sae)
        self.decoder_weight = torch.nn.Parameter(torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [.1, .2, -.1], [.1, -.1, .3],
        ]))
        self.bias = torch.nn.Parameter(torch.tensor([.3, -.2, .1]))

    def encode(self, x):
        return torch.cat([x, torch.ones((*x.shape[:-1], 2), dtype=x.dtype, device=x.device)], -1)

    def decode(self, z):
        return z @ self.decoder_weight + self.bias


def setup_pair(hf=False):
    base = TinyHFHooks() if hf else TinyModel()
    return ModelAdapter(base), AffineSAE(), ActivationSite(2, "blocks.2.hook_resid_post", -2)


@pytest.mark.parametrize("hf", [False, True])
def test_backend_identifies_capabilities(hf):
    model, _, _ = setup_pair(hf)
    assert model.capabilities.forward and model.capabilities.capture and model.capabilities.patch
    assert model.capabilities.generate
    assert model.backend == ("hf_hooks" if hf else "transformer_lens")


@pytest.mark.parametrize("hf", [False, True])
def test_capture_position_and_identity_patch(hf):
    m, sae, site = setup_pair(hf)
    ids = m.tokens(prompt="2 3 4")
    h = m.capture(tokens=ids, site=site)
    assert torch.allclose(h, m.model._hidden(ids)[0, 1])
    m.assert_identity_patch(tokens=ids, site=site)
    snap = capture_features(m, sae, tokens=ids, site=site)
    assert snap.codes.shape == (5,) and snap.reconstruction.shape == (3,)
    assert torch.equal(snap.activation, h)


@pytest.mark.parametrize("hf", [False, True])
def test_edits_only_selected_feature_keep_residual_and_affine_bias(hf):
    m, sae, site = setup_pair(hf)
    sn = capture_features(m, sae, site=site, prompt="2 3 4")
    original_codes = sn.codes.clone()
    replacement = {0: 0.0}
    patch = edit_features(sae, sn, replacements=replacement)
    assert patch.codes[0] == 0
    assert torch.equal(patch.codes[1:], original_codes[1:])
    assert torch.equal(sn.codes, original_codes)
    # A decoder's bias cancels; the remainder h - decode(z) stays present.
    delta = sae.decode(patch.codes[None])[0] - sae.decode(sn.codes[None])[0]
    assert torch.allclose(patch.activation, sn.activation + delta)
    assert torch.allclose(
        patch.activation - sae.decode(patch.codes[None])[0], sn.residual
    )
    zero = edit_features(sae, sn, replacements={})
    assert torch.equal(zero.activation, sn.activation)
    assert torch.equal(zero.codes, sn.codes)


@pytest.mark.parametrize("hf", [False, True])
def test_feature_intervention_changes_downstream_logits(hf):
    m, sae, site = setup_pair(hf)
    out = evaluate_edit(m, sae, site=site, prompt="2 3 4", replacements={0: 0})
    assert out.baseline_logits.shape == (4,)
    assert torch.linalg.vector_norm(out.logit_delta) > 0
    unchanged = evaluate_edit(m, sae, site=site, prompt="2 3 4", replacements={})
    assert torch.allclose(unchanged.logit_delta, torch.zeros(4))


def test_coalition_is_scoped_to_selected_features():
    z = torch.tensor([1., 2., 3., 4.])
    assert coalition_replacements(z, selected_indices=[0, 2], retained_indices=[2]) == {0: 0.0}
    assert coalition_replacements(z, selected_indices=[0, 2], retained_indices=[0, 2]) == {}
    assert coalition_replacements(z, selected_indices=[], retained_indices=[]) == {}
    with pytest.raises(ValueError, match="subset"):
        coalition_replacements(z, selected_indices=[0], retained_indices=[1])
    with pytest.raises(ValueError, match="duplicate"):
        coalition_replacements(z, selected_indices=[1, 1], retained_indices=[])
    with pytest.raises(IndexError):
        coalition_replacements(z, selected_indices=[4], retained_indices=[])


@pytest.mark.parametrize("hf", [False, True])
def test_invalid_site_position_and_hook_are_rejected(hf):
    m, _, site = setup_pair(hf)
    ids = m.tokens(prompt="2 3 4")
    with pytest.raises(IndexError, match="outside"):
        m.capture(tokens=ids, site=ActivationSite(2, site.hook_name, -4))
    with pytest.raises(ValueError, match="belongs"):
        ActivationSite(1, site.hook_name, -1)
    with pytest.raises(ValueError, match="outside model"):
        m.capture(tokens=ids, site=ActivationSite(4, "blocks.4.hook_resid_post", -1))
    if hf:
        with pytest.raises(UnsupportedCapability, match="block output"):
            m.capture(tokens=ids, site=ActivationSite(2, "blocks.2.hook_mlp_out"))


def test_input_validation_and_feature_dimensions():
    m, sae, site = setup_pair()
    with pytest.raises(ValueError, match="exactly one"):
        m.tokens()
    with pytest.raises(ValueError, match="exactly one"):
        m.tokens(prompt="2 3", tokens=torch.ones((1, 2), dtype=torch.long))
    with pytest.raises(TypeError, match="integer"):
        m.tokens(tokens=torch.ones((1, 2)))
    with pytest.raises(ValueError, match="shape"):
        m.tokens(tokens=torch.ones((2, 3), dtype=torch.long))
    sn = capture_features(m, sae, prompt="1 2 3", site=site)
    with pytest.raises(IndexError, match="outside"):
        edit_features(sae, sn, replacements={10: 3})
    with pytest.raises(ValueError, match="finite"):
        edit_features(sae, sn, replacements={1: float("nan")})
    bad_sae = AffineSAE()
    bad_sae.cfg.d_in = 4
    with pytest.raises(ValueError, match="activation"):
        capture_features(m, bad_sae, prompt="1 2 3", site=site)


def test_logit_only_supports_forward_not_patch():
    m = ModelAdapter(lambda ids: torch.zeros((1, ids.shape[1], 3)))
    assert m.backend == "logits_only"
    assert not m.capabilities.capture and not m.capabilities.patch
    ids = torch.tensor([[0, 1]], dtype=torch.long)
    assert m.forward_logits(tokens=ids).shape == (3,)
    with pytest.raises(UnsupportedCapability):
        m.capture(tokens=ids, site=ActivationSite(0, "blocks.0.hook_resid_post"))


def test_identity_replay_detects_wrong_backend_response():
    class IncorrectPatch(TinyHFHooks):
        def run_with_patch(self, input_ids, layer, patches, *, readout_offset=0):
            return super().run_with_patch(input_ids, layer, patches, readout_offset=readout_offset) + 1
    m = ModelAdapter(IncorrectPatch())
    with pytest.raises(RuntimeError, match="identity patch"):
        m.assert_identity_patch(tokens=m.tokens(prompt="1 2 3"),
                                site=ActivationSite(2, "blocks.2.hook_resid_post"))


def test_provenance_records_only_known_revision_and_transfer():
    sae = AffineSAE()
    identity = SAEIdentity.for_custom(
        sae=sae, model_key="tiny", checkpoint="lab/tiny-sae",
        sae_id="layer2", hook_name="blocks.2.hook_resid_post", layer=2,
    )
    assert identity.revision is None
    assert identity.normalization == "unspecified"
    assert identity.d_sae == 5
    assert identity.site == ActivationSite(2, "blocks.2.hook_resid_post")
    assert json.loads(json.dumps(identity.to_dict()))["checkpoint"] == "lab/tiny-sae"
    raw = {
        "model_key": "train", "sae_key": "x", "sae_release": "org/x",
        "sae_id": "layer_2", "hook_name": "blocks.2.hook_resid_post",
        "layer": 2, "loader": "sae_lens", "trained_on_model_key": "train",
        "target_model_key": "deploy", "transfer_kind": "base_to_instruct",
    }
    deploy = SAEIdentity.from_loaded(sae=sae, metadata=raw, model_key="deploy")
    assert deploy.model_key == "deploy"
    assert deploy.trained_on_model_key == "train" and deploy.target_model_key == "deploy"
    with pytest.raises(ValueError, match="not allowed"):
        SAEIdentity.from_loaded(sae=sae, metadata=raw, model_key="other")


def test_model_generate_capability():
    m, _, _ = setup_pair()
    out = m.generate(m.tokens(prompt="1 2"))
    assert out.tolist() == [[1, 2, 1]]


def test_huggingface_datasets_import_works_without_sys_path_mutation(monkeypatch):
    before_path = list(sys.path)
    from models.hf_compat import import_hf_datasets, load_repo_dataset_module
    hf = SimpleNamespace(load_dataset=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "datasets", hf)
    assert import_hf_datasets() is hf
    own = load_repo_dataset_module("SimpleCopying", "SimpleCopying")
    assert hasattr(own, "generate_copying_prompts")
    importlib.import_module("saebase.data.prompt_store")
    assert list(sys.path) == before_path
    assert "saebase.data.prompt_store" in sys.modules
    assert sys.modules["datasets"] is hf


def test_canonical_datasets_tree_and_hf_import_coexist():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "datasets" / "plugins.py").is_file()
    assert not (root / "research_datasets").exists()
    assert not (root / "dlbase").exists()
    assert (root / "saebase" / "data" / "prompt_store.py").exists()
    from saebase.data.artifact_layout import DEFAULT_DATA_ROOT, DEFAULT_EXPERIMENT_ROOT
    assert DEFAULT_DATA_ROOT == root / "data"
    assert DEFAULT_EXPERIMENT_ROOT == root / "experiment"


def test_capture_only_backend_keeps_capabilities_separate():
    class CaptureOnly(TinyModel):
        run_with_hooks = None
    adapter = ModelAdapter(CaptureOnly())
    assert adapter.capabilities.capture and not adapter.capabilities.patch
    h = adapter.capture(
        tokens=adapter.tokens(prompt="1 2"),
        site=ActivationSite(2, "blocks.2.hook_resid_post"),
    )
    assert h.numel() == 3
    with pytest.raises(UnsupportedCapability):
        adapter.patch_logits(tokens=adapter.tokens(prompt="1 2"),
                             site=ActivationSite(2, "blocks.2.hook_resid_post"),
                             activation=h)


def test_real_hf_hooks_wrapper_with_tiny_module_and_offset_replay():
    from models.hf_hooks_backend import HFHooksModel

    class Layer(torch.nn.Module):
        def forward(self, h):
            return h * 1.25 + .25

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=2)
            self.embed = torch.nn.Embedding(20, 3)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.lm_head = torch.nn.Linear(3, 7, bias=False)
            with torch.no_grad():
                self.embed.weight.copy_(torch.arange(60).float().reshape(20, 3) / 100)
                self.lm_head.weight.fill_(.1)

        def forward(self, input_ids):
            h = self.embed(input_ids)
            for layer in self.model.layers:
                h = layer(h)
            return SimpleNamespace(logits=self.lm_head(h))

    class Tok:
        bos_token_id = None
        def encode(self, text, add_special_tokens=False):
            return [int(v) for v in text.split()]

    wrapper = HFHooksModel(TinyCausalLM().eval(), Tok(), model_name="synthetic", device="cpu")
    adapter = ModelAdapter(wrapper)
    site = ActivationSite(0, "blocks.0.hook_resid_post", -2)
    ids = adapter.tokens(prompt="1 2 3")
    assert adapter.capabilities.capture and adapter.capabilities.patch
    adapter.assert_identity_patch(tokens=ids, site=site)
    before = adapter.forward_logits(tokens=ids, readout_position=-2)
    h = adapter.capture(tokens=ids, site=site)
    changed = adapter.patch_logits(tokens=ids, site=site, activation=h + 1,
                                   readout_position=-2)
    assert changed.shape == before.shape
    assert torch.linalg.vector_norm(changed - before) > 0


def test_registered_loader_rejects_hook_drift(monkeypatch):
    from saebase.api import load_registered
    import saebase.sae.loading as loading
    base = TinyModel()
    sae = AffineSAE()
    meta = {
        "model_key":"pythia70m", "sae_key":"pythia70m_resid_sm",
        "sae_release":"pythia-70m-deduped-res-sm", "sae_id":"blocks.3.hook_resid_post",
        "layer":3, "hook_name":"blocks.3.hook_mlp_out", "loader":"sae_lens"
    }
    monkeypatch.setattr(loading, "load_model_and_sae", lambda *a, **kw: (base, sae, meta))
    with pytest.raises(ValueError, match="differs from"):
        load_registered(sae_key="pythia70m_resid_sm", layer=3)


def test_loaded_identity_rejects_metadata_dim_drift():
    sae = AffineSAE()
    meta = {
        "model_key": "train", "sae_key": "x", "sae_release": "org/x",
        "sae_id": "site", "hook_name": "blocks.1.hook_resid_post",
        "layer": 1, "loader": "sae_lens", "d_in": 1000,
    }
    with pytest.raises(ValueError, match="differs from loaded SAE"):
        SAEIdentity.from_loaded(sae=sae, metadata=meta)


def test_registered_loader_fake_backend_complete_path(monkeypatch):
    from saebase.api import load_registered
    import saebase.sae.loading as loading
    base, sae = TinyModel().eval(), AffineSAE().eval()
    base.last_hook = "blocks.3.hook_resid_post"
    metadata = {
        "model_key": "pythia70m", "sae_key": "pythia70m_resid_sm",
        "sae_release": "pythia-70m-deduped-res-sm", "sae_id": "blocks.3.hook_resid_post",
        "layer": 3, "hook_name": base.last_hook, "loader": "sae_lens",
        "d_in": 3, "d_sae": 5, "loaded_model_name":"synthetic-test-model",
    }
    monkeypatch.setattr(loading, "load_model_and_sae", lambda *a, **kw: (base, sae, metadata))
    bundle = load_registered(sae_key="pythia70m_resid_sm", layer=3, model_key="pythia70m")
    assert bundle.identity.loaded_model_name == "synthetic-test-model"
    state = capture_features(bundle.model, bundle.sae, site=bundle.site, prompt="1 2 3")
    assert state.codes.shape == (5,)
    result = evaluate_edit(bundle.model, bundle.sae, site=bundle.site,
                           prompt="1 2 3", replacements={0: 0})
    assert result.logit_delta.abs().sum() > 0
