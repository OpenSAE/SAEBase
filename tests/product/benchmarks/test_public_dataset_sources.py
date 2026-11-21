"""Offline source semantics and unified-entry regression for added text Datasets."""

from conf import load_group

import pytest
from omegaconf import OmegaConf

from saebase.datasets.public_sources import normalize, source_records
from saebase.registry import get_dataset_plugin


@pytest.mark.parametrize(
    "name",
    [
        "HotpotQA",
        "BBH",
        "RAVEL",
        "ARCChallenge",
        "GSM8K",
        "HumanEval",
        "TruthfulQA",
        "HarmBench",
        "XSTest",
    ],
)
def test_registered_capture_policy(name):
    plugin = get_dataset_plugin(name)
    assert callable(plugin.prepare)
    cfg = OmegaConf.create(
        {"dataset": {"name": name}, "model": {"name": "gpt2small"}, "task": {"name": "capture"}}
    )
    plugin.assert_task_supported(cfg)
    cfg.task.name = "precheck"
    plugin.assert_task_supported(cfg)
    cfg.task.name = "train"
    with pytest.raises(ValueError, match="no native protocol"):
        plugin.assert_task_supported(cfg)


def test_gold_answers_not_in_prompts():
    assert normalize("GSM8K", {"question": "How many?", "answer": "SECRET"}) == "How many?\nAnswer:"
    assert (
        normalize("HumanEval", {"prompt": "def f():", "canonical_solution": "SECRET"}) == "def f():"
    )
    prompt = normalize(
        "ARCChallenge",
        {
            "question": "Pick",
            "choices": {"label": ["A", "B"], "text": ["red", "blue"]},
            "answerKey": "B",
        },
    )
    assert prompt.endswith("Answer:") and "Answer: B" not in prompt
    assert (
        normalize("HarmBench", {"FunctionalCategory": "multimodal", "Behavior": "needs image"})
        == ""
    )


def test_cache_integrity_and_preserved_split(monkeypatch, tmp_path):
    import saebase.datasets.public_sources as owner

    monkeypatch.setattr(owner, "default_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        owner,
        "_acquire",
        lambda name, limit: (
            [{"question": "Real source fixture?", "answer": "withheld"}],
            {"revision": "a" * 40, "upstream_split": "test"},
        ),
    )
    records, metadata = source_records("GSM8K", limit=1)
    assert metadata["upstream_split"] == "test"
    assert records[0]["raw"]["answer"] == "withheld"
    assert "withheld" not in records[0]["prompt"]
    source = tmp_path / "sources/GSM8K/prefix_1/records.json"
    source.write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        source_records("GSM8K", limit=1)


def test_scoped_pair_reuse_rejects_wrong_identity():
    from types import SimpleNamespace
    from saebase.runtime.sae_operations import loaded_pair_session, resolve_sae, load_selection

    selection = resolve_sae(
        OmegaConf.create({"name": "gpt2small"}), "gpt2small_res_jb_l8", 8, operation="capture"
    )
    wrong = SimpleNamespace(
        identity=SimpleNamespace(
            model_key="wrong", sae_key=selection.sae_key, layer=8, hook_name=selection.hook
        )
    )
    with loaded_pair_session(wrong):
        with pytest.raises(ValueError, match="identity differs"):
            load_selection(selection, "cpu")


def test_batch_worker_does_real_dataset_validation(tmp_path, monkeypatch):
    from tasks.precheck.operation import _WORKER_MODE, _run_worker

    monkeypatch.chdir(tmp_path)
    token = _WORKER_MODE.set("in_process")
    try:
        # Missing real records cannot become a synthetic PASS in accelerated mode.
        result = _run_worker(
            {
                "kind": "dataset",
                "dataset_key": "GSM8K",
                "model_key": "gpt2small",
                "data_root": str(tmp_path),
            },
            label="dataset_check",
            timeout=10,
            allow_network=False,
        )
        assert result["status"] == "BLOCKED"
        assert result["hard_kill_isolation"] is False
        with pytest.raises(ValueError, match="network isolation"):
            _run_worker({"kind": "sae"}, label="denied", timeout=10, allow_network=False)
    finally:
        _WORKER_MODE.reset(token)


def test_neuronpedia_legacy_hook_is_checked_not_ignored():
    from saebase.api.neuronpedia import _verify_source

    expected = {
        "id": "gpt2-small/8-res-jb",
        "repo": "repo",
        "folder": "layer8",
        "d_in": 768,
        "d_sae": 24576,
        "hook": "blocks.8.hook_resid_pre",
    }
    config = {"d_in": 768, "d_sae": 24576, "hook_point": "blocks.8.hook_resid_pre"}
    source = {
        "modelId": "gpt2-small",
        "id": "8-res-jb",
        "hfRepoId": "repo",
        "hfFolderId": "layer8",
        "saelensConfig": config,
    }
    _verify_source(source, expected)
    config["hook_name"] = "blocks.9.hook_resid_pre"
    with pytest.raises(ValueError, match="disagree"):
        _verify_source(source, expected)


def test_profile_provenance_tracks_selected_layer_and_topk():
    from saebase.api.sae_profiles import get_profile

    profile = get_profile("gemma3_4b_pt_scope_resid")
    meta = profile.transfer_meta(17)
    assert meta["logical_sae_id"] == "layer_17_width_65k_l0_medium"
    assert meta["resolved_physical_path"] == "resid_post/layer_17_width_65k_l0_medium"
    assert get_profile("qwen3_1_7b_base_resid_w32k_l0_100").top_k == 100
    assert get_profile("qwen3_1_7b_base_resid_w32k").top_k == 50


def test_read_only_batch_hashes_detect_mutation():
    import torch
    from types import SimpleNamespace
    from saebase.runtime.sae_operations import loaded_pair_session, pair_state_fingerprints

    pair = SimpleNamespace(
        model=SimpleNamespace(model=torch.nn.Linear(2, 2)), sae=torch.nn.Linear(2, 3)
    )
    first = pair_state_fingerprints(pair)
    with pytest.raises(RuntimeError, match="signature"):
        with loaded_pair_session(pair):
            assert pair_state_fingerprints(pair) == first
            with torch.no_grad():
                pair.sae.weight.add_(1)
            pair_state_fingerprints(pair)
    assert pair_state_fingerprints(pair) != first


def test_r1_uses_native_release_scaling_when_available():
    from saebase.api.sae_loading import _phase_a_sae_lens_source
    from saebase.api.sae_profiles import get_profile

    release, site, converter = _phase_a_sae_lens_source(
        get_profile("r1_distill_llama8b_llamascope_l23"), 23
    )
    assert release == "llama_scope_r1_distill"
    assert site == "l23r_400m_slimpajama_400m_openr1_math"
    assert converter is None


def test_capture_skips_unembedding_preserving_real_transformer_activations(monkeypatch):
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from saebase.api import ModelAdapter, ActivationSite

    # Explicit synthetic random tiny model for software equivalence only.
    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=1,
            d_model=8,
            n_ctx=8,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            d_vocab=20,
            act_fn="relu",
            device="cpu",
        )
    )
    tokens = torch.tensor([[1, 2, 3]])
    _, cache = model.run_with_cache(tokens, names_filter="blocks.0.hook_resid_post")
    expected = cache["blocks.0.hook_resid_post"][0, -1].clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Capture must not allocate discarded vocabulary logits")

    monkeypatch.setattr(model.unembed, "forward", forbidden)
    actual = ModelAdapter(model).capture(
        tokens=tokens, site=ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post")
    )
    assert torch.equal(expected, actual)


def test_hf_prompt_forward_matches_explicit_token_forward():
    import torch
    from types import SimpleNamespace
    from models.hf_logits_backend import HFLogitsModel

    class TinyTokenizer:
        bos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return [2, 3]

    class TinyModel:
        config = SimpleNamespace(num_hidden_layers=1)

        def __call__(self, input_ids):
            return SimpleNamespace(logits=input_ids.float().unsqueeze(-1))

    model = HFLogitsModel(TinyModel(), TinyTokenizer(), model_name="synthetic", device="cpu")
    assert torch.equal(model("fixture"), model(model.to_tokens("fixture")))


def test_neuronpedia_collection_needs_exact_trusted_catalog_binding():
    from saebase.api.neuronpedia import _verify_source

    expected = {
        "id": "model/layer23",
        "repo": "same-repo",
        "folder": "collection/L23R",
        "d_in": 4096,
        "d_sae": 32768,
        "hook": "blocks.23.hook_resid_post",
        "catalog_verified": True,
    }
    source = {
        "modelId": "model",
        "id": "layer23",
        "hfRepoId": "same-repo",
        "hfFolderId": "collection",
        "saelensConfig": None,
    }
    assert (
        _verify_source(source, expected)
        == "official_catalog_binding; remote dimensions unavailable"
    )
    source["hfFolderId"] = "different-collection"
    with pytest.raises(ValueError, match="repo/folder"):
        _verify_source(source, expected)
    source["hfFolderId"] = "collection"
    expected["catalog_verified"] = False
    with pytest.raises(ValueError, match="repo/folder"):
        _verify_source(source, expected)


def test_selected_models_declare_a_resolvable_default_site_without_loading_weights():
    from saebase.runtime.sae_operations import resolve_sae

    selections = {
        "gpt2small": 8,
        "gemma2_2b_base": 25,
        "gemma1b": 13,
        "gemma3_4b_pt": 17,
        "gemma3_12b_it": 24,
        "llama31_8b": 21,
        "qwen3_1_7b_base": 14,
        "qwen3_8b": 18,
        "qwen35_9b": 16,
        "r1_distill_llama8b": 23,
    }
    for key, layer in selections.items():
        model = load_group("model", key).model
        assert resolve_sae(OmegaConf.create(model), operation="capture").layer == layer


def test_prefix_capture_matches_full_forward_and_never_executes_tail(monkeypatch):
    """Synthetic tiny TransformerLens model; exact activation and weight equality."""
    import torch
    from types import SimpleNamespace
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from saebase.api import ActivationSite, ModelAdapter
    from tasks.capture.operation import offload_capture_tail

    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=3,
            d_model=8,
            n_ctx=12,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            d_vocab=20,
            act_fn="relu",
            device="cpu",
        )
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    site = ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post")
    _, cache = model.run_with_cache(tokens, names_filter=site.hook_name)
    expected = cache[site.hook_name][0, -1].clone()
    original = {name: value.clone() for name, value in model.state_dict().items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Unused tail must not execute")

    monkeypatch.setattr(model.blocks[1], "forward", forbidden)
    monkeypatch.setattr(model.unembed, "forward", forbidden)
    adapter = ModelAdapter(model)
    with offload_capture_tail(SimpleNamespace(model=adapter, site=site)):
        actual = adapter.capture(tokens=tokens, site=site)
    assert torch.equal(actual, expected)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
