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
    cfg = OmegaConf.load("conf/tasks/capture/task.yaml")
    assert settings(cfg)["filter"] == settings(cfg)["capture_count"] == 20
    cfg.task.capture.output.feature_count = 3
    assert settings(cfg)["filter"] == 20 and settings(cfg)["capture_count"] == 3
    cfg.task.capture.selection.candidate_count = "all"
    cfg.task.capture.output.feature_count = None
    assert settings(cfg)["capture_count"] == "all"
    assert settings(OmegaConf.create({"task": {"mode": "g_s"}}))["filter"] == 10


@pytest.mark.parametrize("limit,count", [(True, 1), (0, 1), ("10", 1), (2, 3), (3, False)])
def test_invalid_selection_is_rejected(limit, count):
    with pytest.raises(ValueError):
        settings(OmegaConf.create({"task": {"filter": limit, "capture_count": count}}))


@pytest.fixture
def pair(monkeypatch, tmp_path):
    # This fixture isolates feature/annotation behavior; generation is tested separately.
    monkeypatch.setattr(
        "tasks.capture.model_answer.answers",
        lambda cfg, pair, prompts: [
            {"status": "completed", "text": "synthetic continuation"} for _ in prompts
        ],
    )
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
    import saebase.runtime.sae_operations

    monkeypatch.setattr(
        saebase.runtime.sae_operations, "pair_state_fingerprints", lambda _: ("a" * 64, "b" * 64)
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


def test_sar_uses_shared_mask_scorer_without_changing_sampled_responses(
    pair, tmp_path, monkeypatch
):
    from tests.product.capture.test_capture_sar import PAPER_DOUBLE
    from tasks.capture import measure_behavior

    source = tmp_path / "sar.py"
    source.write_text(PAPER_DOUBLE)
    cfg = configuration("sar", 2)
    cfg.task.filter = 3
    cfg.task.sar = {
        "source": str(source),
        "readout": "answer_mean_logprob",
        "answer": "0 1",
        "feature_selection": "native_activation",
        "M": 48,
        "n_fit": 40,
        "feature_filter": {"enabled": False},
        "cohort_filter": {"enabled": False},
    }
    cfg.task.performance = {"enabled": False}
    reference = operation.execute(cfg)["sar"]
    with np.load(reference["artifact"]["path"]) as artifact:
        masks, scores = artifact["masks"].copy(), artifact["responses"].copy()
    calls = []
    original = measure_behavior.score_masks

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(measure_behavior, "score_masks", observed)
    destination = tmp_path / "batched"
    destination.mkdir()
    monkeypatch.chdir(destination)
    cfg.task.performance.enabled = True
    cfg.task.performance.batch_size = 2
    measured = operation.execute(cfg)["sar"]
    assert calls == [True]
    with np.load(measured["artifact"]["path"]) as artifact:
        np.testing.assert_array_equal(artifact["masks"], masks)
        np.testing.assert_allclose(artifact["responses"], scores, rtol=0, atol=1e-7)


@pytest.mark.parametrize("limit,expected", [(3, 3), ("all", 7)])
def test_surrogate_candidate_count_is_also_independent_of_pair_output(
    pair, tmp_path, limit, expected
):
    from tests.product.capture.test_capture_sar import PAPER_DOUBLE

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


def test_i_gs_selection_scores_all_filtered_active_features_before_topk(pair):
    cfg = configuration("I", 3)
    cfg.task.filter = 2
    cfg.task.sar.i_feature_selection = "g_s"
    cfg.task.sar.feature_filter = {"enabled": False}
    cfg.task.sar.cohort_filter = {"enabled": False}
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=cfg.task.prompt)
    active = [i for i in range(7) if float(snapshot.codes[i]) != 0]
    blocked = active[0]
    context = {
        "settings": {"feature_filter": {"enabled": False}, "cohort_filter": {"enabled": False}},
        "blocked_features": {blocked: {}},
        "cohort_blocked_features": set(),
    }
    answer = torch.tensor([[0, 1]])
    expected = {}
    for fid in active:
        if fid == blocked:
            continue
        patch = edit_features(pair.sae, snapshot, replacements={fid: 0.0})
        expected[fid] = float(
            pair.model.answer_token_logprobs(
                prompt_tokens=snapshot.tokens,
                answer_tokens=answer,
                site=pair.site,
                activation=patch.activation,
            )
            .double()
            .mean()
        )
    result = operation._capture(
        cfg, pair=pair, prompt=cfg.task.prompt, snapshot=snapshot, sar_context=context
    )
    measured = result["sar"]
    screening = measured["preselection"]
    assert screening["eligible_pool_count"] == len(expected) > 2
    assert {r["feature_id"] for r in screening["ranking"]} == set(expected)
    for row in screening["ranking"]:
        assert row["g"] == expected[row["feature_id"]]
    selected = measured["feature_ids"]
    assert len(selected) == 2
    assert min(expected[fid] for fid in selected) >= max(
        expected[fid] for fid in expected if fid not in selected
    )
    assert result["candidate_feature_ids"] == selected
    assert measured["measured_states"] == 4
    assert measured["identity"]["behavior"]["feature_selection_uses_answer"] is True
    with np.load(measured["artifact"]["path"]) as saved:
        assert set(saved["screen_feature_ids"].tolist()) == set(expected)
        assert len(saved["screen_deleted_responses"]) == len(expected)


def test_i_ranking_parameter_does_not_change_other_mode_candidates():
    from tasks.capture.selection import candidate_ids

    norm = SimpleNamespace(rank=lambda codes, count: (None, [1, 0, 2]))
    for mode in ["whiten", "non_whiten", "g_s", "sar"]:
        cfg = configuration(mode)
        cfg.task.filter = 2
        cfg.task.sar.i_feature_selection = "g_s"
        codes = torch.tensor([1.0, 2.0, 0.0])
        actual = candidate_ids(cfg, None, codes, norm, None, {})
        cfg.task.sar.i_feature_selection = None
        assert actual == candidate_ids(cfg, None, codes, norm, None, {})


def test_gs_i_selection_requires_a_finite_candidate_count():
    cfg = configuration("I")
    cfg.task.sar.i_feature_selection = "g_s"
    with pytest.raises(ValueError, match=r"integer task\.capture\.selection\.candidate_count"):
        settings(cfg)


@pytest.mark.parametrize("cohort_blocked", [True, False])
def test_i_gs_relaxes_filters_only_when_pool_is_short(pair, cohort_blocked):
    from tasks.capture.selection import candidate_ids

    cfg = configuration("I", 3)
    cfg.task.filter = 3
    cfg.task.sar.i_feature_selection = "g_s"
    cfg.task.sar.i_g_s_shortfall = "relax_filters"
    cfg.task.sar.feature_filter = {"enabled": False}
    cfg.task.sar.cohort_filter = {"enabled": False}
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=cfg.task.prompt)
    active = [i for i in range(7) if float(snapshot.codes[i]) != 0]
    blocked = set(active[2:])
    context = {
        "settings": {"feature_filter": {"enabled": False}, "cohort_filter": {"enabled": False}},
        "blocked_features": {} if cohort_blocked else {i: {} for i in blocked},
        "cohort_blocked_features": blocked if cohort_blocked else set(),
    }
    actual = candidate_ids(cfg, pair, snapshot.codes, None, context, {})
    assert set(actual) == set(active)
    assert context["i_g_s_pool"]["fully_filtered_count"] == 2
    assert context["i_g_s_pool"]["relaxed_filters"] == (
        ["cohort_frequency"] if cohort_blocked else ["semantic_and_native_density"]
    )
    assert context["cohort_blocked_features"] == (blocked if cohort_blocked else set())
    cfg.task.filter = 2
    actual = candidate_ids(cfg, pair, snapshot.codes, None, context, {})
    assert set(actual) == set(active[:2])
    assert context["i_g_s_pool"]["relaxed_filters"] == []


def test_g_s_whiten_pool_then_independent_deletion_scores(pair):
    from tasks.capture.whitening import Normalization

    cfg = configuration("g_s", 2)
    cfg.task.capture = {
        "g_s": {"candidate_count": 3},
        "selection": {"feature_filter": {"enabled": False}, "cohort_filter": {"enabled": False}},
    }
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=cfg.task.prompt)
    scores = []
    for fid in range(7):
        edited = edit_features(pair.sae, snapshot, replacements={fid: 0.0})
        scores.append(
            float(
                pair.model.answer_token_logprobs(
                    prompt_tokens=snapshot.tokens,
                    answer_tokens=torch.tensor([[0, 1]]),
                    site=pair.site,
                    activation=edited.activation,
                )
                .double()
                .mean()
            )
        )
    best = int(np.argmax(scores))
    others = [fid for fid in range(7) if fid != best]
    order = [*others[:2], best, *others[2:]]
    z = np.empty(7)
    for index, fid in enumerate(order):
        z[fid] = 10 - index
    norm = Normalization(snapshot.codes.numpy() - z, np.ones(7), np.ones(7), {}, {})
    result = operation._capture(
        cfg, pair=pair, prompt=cfg.task.prompt, snapshot=snapshot, normalization=norm
    )
    assert result["candidate_feature_ids"] == order[:3]
    assert result["g_s"]["measured_states"] == 4
    assert result["top_features"][0]["feature_id"] == best
    assert result["top_features"][0]["g"] == scores[best]
    assert len(result["top_features"]) == 2
    assert result["g_s"]["preselection"]["method"] == "absolute_zscore"
    assert result["g_s"]["preselection"]["requested_count"] == 3
    cfg.task.capture.g_s.candidate_count = 5
    widened = operation._capture(
        cfg, pair=pair, prompt=cfg.task.prompt, snapshot=snapshot, normalization=norm
    )
    assert widened["candidate_feature_ids"] == order[:5]
    assert widened["g_s"]["measured_states"] == 6
    assert len(widened["top_features"]) == 2


def test_new_g_s_defaults_and_mode_isolation():
    from hydra import compose, initialize_config_module
    from tasks.capture.selection import needs_normalization
    from tasks.capture.task import CaptureTask

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="config", overrides=["task=capture", "task.capture.mode=g_s"])
    assert settings(cfg)["filter"] == settings(cfg)["capture_count"] == 20
    normalized = CaptureTask().continuation_configuration({"task": cfg.task})
    assert "g_s" not in normalized["task"]["capture"]
    cfg.task.capture.selection.candidate_policy = "legacy"
    assert settings(cfg)["filter"] == 100
    assert settings(cfg)["capture_count"] == 10
    assert needs_normalization(cfg)
    assert cfg.task.capture.i.feature_selection == "whiten"
    cfg.task.capture.g_s.candidate_count = 50
    assert settings(cfg)["filter"] == 50 and settings(cfg)["capture_count"] == 10
    for mode in ["whiten", "non_whiten", "I", "sar"]:
        cfg.task.capture.mode = mode
        assert settings(cfg)["filter"] == 20
        normalized = CaptureTask().continuation_configuration({"task": cfg.task})
        assert "g_s" not in normalized["task"]["capture"]
    cfg.task.capture.mode = "g_s"
    normalized = CaptureTask().continuation_configuration({"task": cfg.task})
    assert normalized["task"]["capture"]["g_s"]["candidate_count"] == 50


@pytest.mark.parametrize("mode", ["non_whiten", "whiten", "g_s", "I", "sar"])
def test_shared_candidates_filter_before_whiten_without_padding(mode):
    from tasks.capture.selection import candidate_ids, needs_normalization
    from tasks.capture.whitening import Normalization

    cfg = OmegaConf.load("conf/tasks/capture/task.yaml")
    cfg.task.capture.mode = mode
    cfg.task.capture.i.feature_selection = "g_s"
    codes = torch.tensor([2.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0])
    scores = np.array([90, 100, 80, -7, 5, 99, 5])
    std = np.array([1, 1, 1, 1, 1, 0, 1])
    normalization = Normalization(codes.numpy() - scores, std, std, {}, {})
    context = {
        "settings": {"feature_filter": {"enabled": True}, "cohort_filter": {"enabled": True}},
        "blocked_features": {0: {}},
        "cohort_blocked_features": {2},
    }
    assert needs_normalization(cfg, context)
    assert candidate_ids(cfg, None, codes, normalization, context, {}) == [3, 4, 6]
    cfg.task.capture.selection.candidate_count = 2
    assert candidate_ids(cfg, None, codes, normalization, context, {}) == [3, 4]


def test_shared_pair_candidate_shortfall_keeps_model_answer(pair):
    from tasks.capture.whitening import Normalization

    cfg = configuration("I")
    cfg.task.capture = {"selection": {"candidate_policy": "active_whiten", "candidate_count": 10}}
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=cfg.task.prompt)
    fid = int(snapshot.codes.abs().argmax())
    normalization = Normalization(np.zeros(7), np.ones(7), np.ones(7), {}, {})
    context = {
        "settings": {"feature_filter": {"enabled": False}, "cohort_filter": {"enabled": False}},
        "blocked_features": dict.fromkeys(set(range(7)) - {fid}, {}),
    }
    result = operation._capture(
        cfg, snapshot=snapshot, normalization=normalization, sar_context=context
    )
    assert result["candidate_feature_ids"] == [fid]
    assert result["completion"]["counts"]["undefined"] == 1
    assert "found 1" in result["candidate_shortfall"]
    assert result["model_answer"]["status"] == "completed"
    assert "sar" not in result


@pytest.mark.parametrize("value", [0, -1, True, "100", None, "all"])
def test_invalid_g_s_whiten_pool_rejected(value):
    cfg = configuration("g_s", 2)
    cfg.task.capture = {"g_s": {"candidate_count": value}}
    with pytest.raises(ValueError, match="g_s.candidate_count"):
        settings(cfg)
