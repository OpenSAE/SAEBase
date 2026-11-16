"""Capture filter/output separation and measured answer behavior contracts."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from saebase.api import ModelAdapter, SAEIdentity, capture_features, edit_features
from tasks.capture import operation
from tasks.capture.selection import settings
from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE


def test_shipped_defaults_and_separate_output_count():
    cfg = OmegaConf.load("conf/task/capture.yaml")
    assert settings(cfg)["filter"] == settings(cfg)["capture_count"] == 10
    cfg.task.capture_count = 3
    assert settings(cfg)["filter"] == 10 and settings(cfg)["capture_count"] == 3
    cfg.task.filter = "all"
    cfg.task.capture_count = None
    assert settings(cfg)["capture_count"] == "all"
    assert settings(OmegaConf.create({"task": {"mode": "g_s"}}))["filter"] == 10


@pytest.mark.parametrize("limit,count", [(True, 1), (0, 1), ("10", 1), (2, 3), (3, False)])
def test_invalid_selection_is_rejected(limit, count):
    with pytest.raises(ValueError):
        settings(OmegaConf.create({"task": {"filter": limit, "capture_count": count}}))


@pytest.fixture
def pair(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    raw = ResearchToyModel().eval()
    raw.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: [int(x) for x in text.split()]
    )
    pair = SimpleNamespace(model=ModelAdapter(raw), sae=ResearchToySAE().eval())
    pair.identity = SAEIdentity(
        "synthetic", "synthetic", "synthetic", "synthetic", raw.site_name, 0, "synthetic", 4, 7
    )
    pair.site = pair.identity.site
    monkeypatch.setattr(operation, "_pair", lambda _: pair)
    import tasks.sae_runtime

    monkeypatch.setattr(
        tasks.sae_runtime, "pair_state_fingerprints", lambda _: ("a" * 64, "b" * 64)
    )
    return pair


def configuration(mode, count=2):
    return OmegaConf.create(
        {
            "seed": 42,
            "dataset": {"name": "explicit"},
            "task": {
                "mode": mode,
                "prompt": "1 2 3",
                "filter": "all",
                "capture_count": count,
                "neuronpedia": {"enabled": False},
                "sar": {
                    "algorithm": "surrogate",
                    "readout": "answer_mean_logprob",
                    "answer": "0 1",
                },
            },
        }
    )


def test_all_raw_features_and_independent_capture_count(pair):
    cfg = configuration("non_whiten", "all")
    result = operation.execute(cfg)
    assert len(result["top_features"]) == result["feature_count"] == 7
    cfg.task.capture_count = 2
    limited = operation.execute(cfg)
    assert len(limited["top_features"]) == 2
    assert limited["candidate_feature_ids"] == result["candidate_feature_ids"]


def test_numeric_filter_applies_existing_blocked_ids_before_output_limit(pair):
    cfg = configuration("non_whiten", 1)
    cfg.task.filter = 3
    cfg.task.sar.feature_filter = {"enabled": False}
    cfg.task.sar.cohort_filter = {"enabled": False}
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=cfg.task.prompt)
    blocked = int(snapshot.codes.abs().argmax())
    context = {
        "settings": {"feature_filter": {"enabled": False}, "cohort_filter": {"enabled": False}},
        "blocked_features": {blocked: {}},
        "cohort_blocked_features": set(),
    }
    result = operation._capture(
        cfg, pair=pair, prompt=cfg.task.prompt, snapshot=snapshot, sar_context=context
    )
    assert len(result["candidate_feature_ids"]) == 3
    assert len(result["top_features"]) == 1
    assert blocked not in result["candidate_feature_ids"]


def test_single_is_best_deletion_not_best_retained_single(pair):
    cfg = configuration("g_s")
    cfg.task.sar.answer_backend = "full_context_tail"
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=cfg.task.prompt)
    scores = []
    answer = torch.tensor([[0, 1]])
    for fid in range(7):
        patch = edit_features(pair.sae, snapshot, replacements={fid: 0.0})
        scores.append(
            float(
                pair.model.answer_token_logprobs(
                    prompt_tokens=snapshot.tokens,
                    answer_tokens=answer,
                    site=pair.site,
                    activation=patch.activation,
                )
                .double()
                .mean()
            )
        )
    result = operation.execute(cfg)
    assert result["g_s"]["measured_states"] == 8
    best = result["top_features"][0]
    assert best["feature_id"] == int(np.argmax(scores))
    assert best["g"] == max(scores)
    assert best["delta_g"] == best["g"] - result["g_s"]["full_response"]
    assert len(result["top_features"]) == 2


def test_exhaustive_pairs_match_independent_cube_differences(pair):
    cfg = configuration("I", 3)
    result = operation.execute(cfg)
    measured = result["sar"]
    with np.load(measured["artifact"]["path"]) as values:
        cube = values["responses"].reshape((2,) * 7)
    maximum = max(
        float(np.diff(np.diff(cube, axis=i), axis=j).max())
        for i in range(7)
        for j in range(i + 1, 7)
    )
    pairs = measured["pairwise"]["top_pairs"]
    assert len(pairs) == 3 and measured["measured_states"] == 128
    assert pairs[0]["interaction"] == pytest.approx(maximum, abs=1e-12)
    assert all(len(item["features"]) == 2 for item in pairs)
    assert measured["pairwise"]["complete_four_state_backgrounds"] == 21 * 32


def test_pair_output_count_can_exceed_candidate_feature_count(pair):
    cfg = configuration("I", 5)
    cfg.task.filter = 4
    cfg.task.sar.feature_selection = "native_activation"
    cfg.task.sar.feature_filter = {"enabled": False}
    cfg.task.sar.cohort_filter = {"enabled": False}
    result = operation.execute(cfg)
    assert len(result["candidate_feature_ids"]) == 4
    assert len(result["sar"]["pairwise"]["top_pairs"]) == 5


@pytest.mark.parametrize("limit,expected", [(3, 3), ("all", 7)])
def test_surrogate_candidate_count_is_also_independent_of_pair_output(
    pair, tmp_path, limit, expected
):
    from tests.product.test_capture_sar import PAPER_DOUBLE

    source = tmp_path / "sar.py"
    source.write_text(PAPER_DOUBLE)
    cfg = configuration("sar", 1)
    cfg.task.filter = limit
    cfg.task.sar = {
        "algorithm": "surrogate",
        "source": str(source),
        "readout": "answer_mean_logprob",
        "answer": "0 1",
        "feature_selection": "native_activation",
        "M": 48,
        "n_fit": 40,
        "feature_filter": {"enabled": False},
        "cohort_filter": {"enabled": False},
    }
    result = operation.execute(cfg)
    assert result["sar"]["identity"]["K"] == expected
    assert len(result["candidate_feature_ids"]) == expected
    assert len(result["sar"]["pairwise"]["top_pairs"]) == 1


@pytest.mark.parametrize("mode", ["g_s", "I"])
def test_dataset_capture_reference_and_output_counts_remain_independent(pair, monkeypatch, mode):
    from saebase.datasets.selection import SampleSelection
    from saebase.results import EvaluationResult
    from tasks.capture.task import CaptureTask

    rows = [
        {"sample_id": str(i), "prompt": f"1 2 {i + 3}", "reference_answer": "0 1"} for i in range(3)
    ]
    requests = []

    def selected(dataset, **kwargs):
        requests.append(dataset.sample_count)
        chosen = rows[: dataset.sample_count]
        return SampleSelection(chosen, {"selected_count": len(chosen)})

    monkeypatch.setattr("saebase.datasets.selection.select_dataset", selected)
    cfg = configuration(mode, 1)
    cfg.model = {"name": "synthetic"}
    cfg.task.name = "capture"
    cfg.task.prompt = ""
    cfg.dataset.sample_count = 1
    cfg.task.filter = 2
    cfg.task.sar.feature_selection = "whiten"
    cfg.task.sar.feature_filter = {"enabled": False}
    cfg.task.sar.cohort_filter = {
        "enabled": True,
        "reference_sample_count": 3,
        "zscore_threshold": 1000.0,
    }
    payload = operation.execute(cfg)
    assert requests == [1, 3]
    assert payload["sample_count"] == 1 and payload["counts"]["completed"] == 1
    result = EvaluationResult.from_payload("capture", cfg, payload)
    assert CaptureTask().continuation_action(result) == "skip"


def test_full_context_tail_matches_native_complete_forward():
    tl = pytest.importorskip("transformer_lens")
    from saebase.api import ActivationSite
    from tasks.capture.answer import prepare_answer_readout, answer_patch

    raw = tl.HookedTransformer(
        tl.HookedTransformerConfig(
            n_layers=2,
            d_model=8,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            n_ctx=16,
            d_vocab=17,
            act_fn="relu",
            device="cpu",
        )
    ).eval()
    raw.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: [int(v) for v in text.split()], padding_side="right"
    )
    adapter = ModelAdapter(raw)
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    tokens = torch.tensor([[1, 2, 3]])
    activation = adapter.capture(tokens=tokens, site=site)

    class IdentitySAE(torch.nn.Module):
        cfg = SimpleNamespace(d_in=8, d_sae=8)

        def encode(self, activation):
            return activation

        def decode(self, codes):
            return codes

    snapshot = SimpleNamespace(
        tokens=tokens, site=site, activation=activation, codes=activation, ln_mu=None, ln_std=None
    )
    pair = SimpleNamespace(model=adapter, sae=IdentitySAE())
    readout = {"reference": {"text": "4 5", "prefix": " "}}
    score, _, readout_activation = prepare_answer_readout(
        pair, snapshot, "1 2 3", readout, feature_ids=[0, 1], answer_backend="full_context_tail"
    )
    patched = activation + 0.25
    native = adapter.answer_token_logprobs(
        prompt_tokens=tokens,
        answer_tokens=torch.tensor([[4, 5]]),
        site=site,
        activation=answer_patch(activation, readout_activation, patched),
    )
    assert score(patched) == pytest.approx(float(native.double().mean()), abs=1e-7)
    assert readout["execution"] == "full_context_tail"


def test_all_whiten_still_uses_separate_reference_cohort(pair, monkeypatch):
    from saebase.datasets.selection import SampleSelection
    from tasks.capture.selection import reference_rows

    cfg = configuration("whiten", 1)
    cfg.dataset.sample_count = 1
    cfg.task.sar.cohort_filter = {"enabled": False, "reference_sample_count": 3}
    rows = [{"sample_id": str(i), "prompt": f"1 2 {i}"} for i in range(3)]
    requested = []

    def select(dataset, **kwargs):
        requested.append(dataset.sample_count)
        return SampleSelection(rows, {"selected_count": 3})

    monkeypatch.setattr("saebase.datasets.selection.select_dataset", select)
    assert reference_rows(cfg, pair, rows[:1], None) == rows
    assert requested == [3]
