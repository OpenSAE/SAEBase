"""Whiten semantics with explicitly synthetic token codes; no downloads or GPU."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from saebase.api import ActivationSite, ModelAdapter
from tasks.capture.whitening import Moments, Normalization


def test_population_moments_weight_tokens_not_prompts():
    rows = torch.tensor([[0.0, 0.0, 3.0], [2.0, 4.0, 3.0], [4.0, 0.0, 3.0]])
    moments = Moments()
    moments.update(rows[:2])
    moments.update(rows[2:])
    assert moments.count == 3
    np.testing.assert_allclose(moments.mean.numpy(), rows.double().numpy().mean(0), atol=1e-14)
    np.testing.assert_allclose((moments.m2 / 3).numpy(), rows.double().numpy().var(0), atol=1e-14)
    assert moments.mean[0] == 2 and moments.m2[-1] == 0


def test_abs_zscore_ties_zero_native_and_zero_variance():
    norm = Normalization(
        np.array([1.0, 1.0, 5.0, 0.0]),
        np.array([0.5, 0.5, 0.0, 1.0]),
        np.array([0.25, 0.25, 0.0, 1.0]),
        {},
        {},
    )
    score, ids = norm.rank(torch.tensor([0.0, 2.0, 5.0, 0.0]), 10)
    assert ids == [0, 1, 3]
    np.testing.assert_array_equal(score, [-2.0, 2.0, 0.0, 0.0])
    assert norm.rank(torch.tensor([0.0, 2.0, 5.0, 0.0]), 1)[1] == [0]


@pytest.mark.parametrize("mode", ["bad", None, True, ["whiten"]])
def test_invalid_mode_rejected_before_loading(mode, monkeypatch):
    import tasks.capture.operation as owner

    monkeypatch.setattr(owner, "_pair", lambda cfg: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="task.mode"):
        owner.execute(OmegaConf.create({"task": {"mode": mode}}))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    import tasks.capture.operation as operation
    import tasks.sae_runtime as runtime
    import saebase.datasets.selection as selection

    monkeypatch.chdir(tmp_path)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))
            self.cfg = SimpleNamespace(d_in=3, d_sae=3)

        def encode(self, x):
            return x @ self.weight.T

        def decode(self, z):
            return z @ self.weight

    class Model:
        model = None
        backend = "synthetic"
        sequence_calls = 0
        capture_calls = 0

        def tokens(self, *, prompt=None, tokens=None):
            return (
                tokens
                if tokens is not None
                else torch.tensor([[1, 2]] if prompt == "p1" else [[3]])
            )

        def capture_sequence(self, *, tokens, site):
            self.sequence_calls += 1
            return (
                torch.tensor([[0.0, 0.0, 3.0], [2.0, 4.0, 3.0]])
                if tokens.shape[1] == 2
                else torch.tensor([[4.0, 0.0, 3.0]])
            )

        def capture(self, *, tokens, site):
            self.capture_calls += 1
            return self.capture_sequence(tokens=tokens, site=site)[-1]

    pair = SimpleNamespace(
        model=Model(),
        sae=Encoder(),
        site=ActivationSite(0, "blocks.0.hook_resid_post"),
        identity=SimpleNamespace(
            model_key="synthetic",
            sae_key="synthetic",
            loaded_model_name="synthetic",
            layer=0,
            d_sae=3,
        ),
    )
    rows = [{"sample_id": "first", "prompt": "p1"}, {"sample_id": "second", "prompt": "p2"}]
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    monkeypatch.setattr(runtime, "loaded_pair_session", lambda pair: nullcontext())
    monkeypatch.setattr(runtime, "pair_state_fingerprints", lambda pair: ("a" * 64, "b" * 64))
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(rows, {"selected_count": 2}),
    )
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {"name": "synthetic"},
            "dataset": {"name": "synthetic", "sample_count": 500},
            "task": {
                "name": "capture",
                "mode": "whiten",
                "whiten_chunk_size": 1,
                "prompt": "",
                "top_feature_num": 10,
                "neuronpedia": {"enabled": False},
            },
        }
    )
    return cfg, pair


def test_full_capture_fits_all_tokens_reuses_forward_and_preserves_raw(corpus, tmp_path):
    from tasks.capture.operation import execute
    from saebase.results import EvaluationResult

    cfg, pair = corpus
    result = execute(cfg)
    assert result["mode"] == "whiten" and result["sample_count"] == 2
    assert result["completion"]["status"] == "complete"
    assert result["normalization"]["identity"]["token_count"] == 3
    with np.load(tmp_path / "metadata/capture_normalization.npz") as stats:
        np.testing.assert_allclose(stats["mean"], [2.0, 4 / 3, 3.0], atol=1e-14)
        np.testing.assert_allclose(stats["variance"], [8 / 3, 32 / 9, 0.0], atol=1e-14)
        assert stats["mean"].dtype == np.float64
    ledger = [
        json.loads(line)
        for line in (tmp_path / "reports/capture_samples.jsonl").read_text().splitlines()
    ]
    second = ledger[1]["result"]
    zero = next(f for f in second["top_features"] if f["feature_id"] == 1)
    assert zero["activation"] == 0 and zero["zero_native_activation"] and zero["zscore"] < 0
    assert all(f["feature_id"] != 2 for f in second["top_features"])
    assert second["top_features"][0]["activation"] == 4
    assert pair.model.sequence_calls == 2 and pair.model.capture_calls == 0
    assert EvaluationResult.from_payload("capture", cfg, result).exit_code == 0
    assert len(result["required_artifacts"]) == 3


def test_explicit_query_uses_the_full_dataset_population(corpus):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    cfg.task.prompt = "p2"
    result = execute(cfg)
    assert result["sample_count"] == 1 and result["normalization"]["prompt_count"] == 2
    assert result["normalization"]["identity"]["token_count"] == 3
    assert pair.model.capture_calls == 1


def test_non_whiten_default_is_compatible_but_whiten_is_distinct():
    from tasks.capture.task import CaptureTask

    task = CaptureTask()
    old = {"task": {"name": "capture", "top_feature_num": 10}}
    new = {
        "task": {
            "name": "capture",
            "top_feature_num": 10,
            "mode": "non_whiten",
            "whiten_chunk_size": 256,
        }
    }
    assert task.continuation_configuration(old) == task.continuation_configuration(new)
    new["task"]["mode"] = "whiten"
    assert task.continuation_configuration(old) != task.continuation_configuration(new)
    assert "mode" not in old["task"]


def test_sequence_api_matches_full_transformerlens_cache():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=2,
            d_model=8,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            n_ctx=8,
            d_vocab=20,
            act_fn="relu",
            device="cpu",
        )
    )
    tokens = torch.tensor([[1, 2, 3]])
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    _, cache = model.run_with_cache(tokens, names_filter=site.hook_name)
    adapter = ModelAdapter(model)
    actual = adapter.capture_sequence(tokens=tokens, site=site)
    assert torch.equal(actual, cache[site.hook_name][0])
    assert torch.equal(actual[-1], adapter.capture(tokens=tokens, site=site))


def test_failed_population_does_not_publish_partial_normalization(corpus, tmp_path, monkeypatch):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    original = pair.model.capture_sequence

    def fail_second(*, tokens, site):
        if tokens.shape[1] == 1:
            raise RuntimeError("synthetic population capture failure")
        return original(tokens=tokens, site=site)

    monkeypatch.setattr(pair.model, "capture_sequence", fail_second)
    result = execute(cfg)
    assert result["status"] == "failed" and result["captured_count"] == 0
    assert result["completion"]["requested"] == 2 and result["completion"]["missing"] == 2
    assert not (tmp_path / "metadata/capture_normalization.npz").exists()


def test_all_constant_features_are_mathematically_undefined(corpus, tmp_path, monkeypatch):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    monkeypatch.setattr(
        pair.model, "capture_sequence", lambda *, tokens, site: torch.ones(tokens.shape[1], 3)
    )
    result = execute(cfg)
    rows = [
        json.loads(line)["result"]
        for line in (tmp_path / "reports/capture_samples.jsonl").read_text().splitlines()
    ]
    assert result["normalization"]["valid_feature_count"] == 0
    assert all(row["top_features"] == [] and row["ranking_status"] == "undefined" for row in rows)
    assert all(row["completion"]["counts"]["undefined"] == 1 for row in rows)


def test_hf_sequence_offsets_preserve_token_order():
    class HF:
        cfg = SimpleNamespace(device="cpu", n_layers=1)

        def get_resid(self, tokens, layer, offsets):
            hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            return torch.stack([hidden[len(hidden) - 1 - offset] for offset in offsets])

    adapter = ModelAdapter(HF(), backend="hf_hooks")
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    sequence = adapter.capture_sequence(tokens=torch.tensor([[1, 2, 3]]), site=site)
    assert torch.equal(sequence, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_cross_token_competition_requires_a_separate_protocol():
    from saebase.api import encode_activations, UnsupportedCapability

    sae = SimpleNamespace(
        cfg=SimpleNamespace(d_in=2, d_sae=2, architecture="batch_topk"),
        encode=lambda h: pytest.fail("must reject before encode"),
        decode=lambda z: z,
    )
    with pytest.raises(UnsupportedCapability, match="tokenwise"):
        encode_activations(sae, torch.ones(2, 2))


def test_modes_are_queryable_side_by_side_without_rewriting_history(tmp_path):
    from saebase.results import EvaluationResult
    from saebase.runtime.experiment_results import record_run, list_runs

    for mode in ["non_whiten", "whiten"]:
        cfg = OmegaConf.create(
            {
                "task": {"name": "capture"},
                "dataset": {"name": "HotpotQA"},
                "model": {"name": "llama31_8b"},
                "seed": 42,
            }
        )
        if mode == "whiten":
            cfg.task.mode = mode
        record_run(
            tmp_path / mode,
            cfg,
            result=EvaluationResult.from_payload("capture", cfg, {"status": "executed"}),
        )
    old = tmp_path / "non_whiten/metadata/manifest.json"
    before = old.read_bytes()
    for mode in ["non_whiten", "whiten"]:
        records = list_runs(
            tmp_path, task="capture", dataset="HotpotQA", model="llama31_8b", task_mode=mode
        )
        assert len(records) == 1 and Path(records[0]["path"]).name == mode
    assert old.read_bytes() == before
