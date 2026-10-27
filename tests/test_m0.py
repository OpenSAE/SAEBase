"""M0 minimal-install contracts and reference recorder tests (no model downloads)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import SAEIdentity, ActivationSite, LoadedResearchPair, ModelAdapter
from saebase.runtime import required_packages, verify_runtime
from saebase.tools.m0_reference_smoke import _sha_weights, compare_references, record_reference

ROOT = Path(__file__).resolve().parents[1]


def make_bundle():
    model = ResearchToyModel().eval()
    sae = ResearchToySAE().eval()
    site = ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post")
    identity = SAEIdentity.for_custom(
        sae=sae, model_key="research-toy", checkpoint="synthetic://test",
        hook_name=site.hook_name, layer=site.layer,
    )
    return LoadedResearchPair(model=ModelAdapter(model), sae=sae, identity=identity)


def test_optional_checkpoint_packages_not_imported_by_base_api():
    code = r'''
import builtins
import torch
base = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'huggingface_hub','safetensors','sae_lens','transformer_lens','transformers'}:
        raise ImportError('optional dependency was eagerly imported: ' + name)
    return base(name, *args, **kwargs)
builtins.__import__ = guarded
import saebase.sae.loading
from saebase.api import capture_features, evaluate_edit
print('minimal imports OK')
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "minimal imports OK" in result.stdout


def test_environment_dependency_layers(monkeypatch):
    core = [name for name, _ in required_packages("core")]
    analysis = [name for name, _ in required_packages("analysis")]
    generation = [name for name, _ in required_packages("generation")]
    assert set(core) == {"torch", "numpy", "hydra", "omegaconf"}
    assert set(core).issubset(analysis)
    assert {"sae_lens", "huggingface_hub", "safetensors"}.issubset(analysis)
    assert "vllm" not in analysis and "vllm" in generation
    monkeypatch.setattr("saebase.runtime.missing_packages", lambda mode: [("fake", "example")])
    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[analysis\]"'):
        verify_runtime(mode="analysis")


def test_model_and_sae_state_fingerprints_are_deterministic_and_sensitive():
    bundle = make_bundle()
    h1 = _sha_weights(bundle.model.model)
    h2 = _sha_weights(bundle.model.model)
    assert h1 == h2 and len(h1) == 64
    with __import__("torch").no_grad():
        first = next(bundle.model.model.parameters())
        first.add_(1)
    assert _sha_weights(bundle.model.model) != h1
    assert len(_sha_weights(bundle.sae)) == 64


def test_reference_measurements_are_real_operations_but_marked_synthetic():
    bundle = make_bundle()
    prompts = [
        {"id": "a", "prompt": "1 2 3", "tokenization": {"input_ids": [1, 2, 3]}},
        {"id": "b", "prompt": "2 3 4", "tokenization": {"input_ids": [2, 3, 4]}},
    ]
    one = record_reference(bundle, prompts, seed=7)
    two = record_reference(bundle, prompts, seed=7)
    assert one["reference_type"] == "synthetic_test"
    assert one["real_checkpoint_validated"] is False
    assert one["status"] == "passed" and one["sample_count"] == 2
    assert one["max_identity_error"] <= 1e-4
    assert all(row["codes_l0"] > 0 for row in one["samples"])
    assert all(row["edit_logit_delta_l2"] > 0 for row in one["samples"])
    assert "1 2 3" not in json.dumps(one)
    compare_references(two, json.loads(json.dumps(one)))
    mutated = json.loads(json.dumps(two))
    mutated["samples"][0]["activation_probe"][0] += 1
    with pytest.raises(RuntimeError, match="numerical drift"):
        compare_references(mutated, one)
    mutated = json.loads(json.dumps(two))
    mutated["model_state_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="model_state_sha256"):
        compare_references(mutated, one)


def test_reference_rejects_token_drift_and_empty_data():
    bundle = make_bundle()
    with pytest.raises(ValueError, match="At least one prepared prompt"):
        record_reference(bundle, [], seed=1)
    bad = [{"prompt": "1 2 3", "tokenization": {"input_ids": [1, 3, 2]}}]
    with pytest.raises(RuntimeError, match="tokenization differs"):
        record_reference(bundle, bad, seed=1)


def test_m0_real_runner_documents_explicit_inputs():
    p = subprocess.run(
        [sys.executable, "saebase/tools/m0_reference_smoke.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0, p.stderr
    assert "--reuse-prompts" in p.stdout
    assert "--compare-to" in p.stdout
    assert (ROOT / "saebase/tools/m0_real_smoke.sh").is_file()


def test_reference_hash_supports_zero_dimensional_state_tensors():
    import torch
    class ScalarState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scalar", torch.tensor(0.5))
            self.register_buffer("vector", torch.arange(5.0))
    module = ScalarState()
    old_hash = _sha_weights(module)
    assert len(old_hash) == 64
    module.scalar += 1
    assert _sha_weights(module) != old_hash


def test_transformerlens_keyword_hook_callback_compatibility():
    import torch

    class KeywordHookModel(ResearchToyModel):
        def run_with_hooks(self, tokens, fwd_hooks):
            h = self._hidden(tokens)
            for name, fn in fwd_hooks:
                assert name == self.site_name
                h = fn(h, hook=None)
            return self._logits(h)

    model = ModelAdapter(KeywordHookModel().eval())
    site = ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post")
    tokens = model.tokens(prompt="1 2 3")
    model.assert_identity_patch(tokens=tokens, site=site)
