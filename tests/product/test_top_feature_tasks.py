"""Top N task comparisons: real synthetic capture ranking, independent edits and saved identity."""

import json
from contextlib import nullcontext

import pytest
from omegaconf import OmegaConf

from tasks.capture.top_features import capture_top_cohort, validate_top_selection

pytest_plugins = ["tests.product.test_capture_whitening"]


@pytest.mark.parametrize("mode", ["non_whiten", "whiten"])
def test_top_features_match_capture_exactly(corpus, mode):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    cfg.task.mode = mode
    cfg.task.top_feature_num = 2
    execute(cfg)
    original = [json.loads(line)["result"] for line in open("reports/capture_samples.jsonl")]
    cfg.task.feature_selection = "capture_top"
    cfg.task.feature_ranking = mode
    selected = list(capture_top_cohort(cfg, pair))
    assert [r["top_features"] for r in selected] == [r["top_features"] for r in original]
    assert all(r["input_identity"]["model_sha256"] == "a" * 64 for r in selected)


@pytest.mark.parametrize("operation", ["intervene", "generate"])
def test_each_feature_is_a_fresh_single_edit(corpus, monkeypatch, operation):
    from tasks.intervene.top_features import run_top_features

    cfg, pair = corpus
    cfg.task.name = operation
    cfg.task.feature_selection = "capture_top"
    cfg.task.feature_ranking = "non_whiten"
    cfg.task.top_feature_num = 2
    cfg.task.intervention = "ablate"
    cfg.task.value = 0.0
    monkeypatch.setattr("tasks.sae_runtime._pair", lambda cfg: pair)
    monkeypatch.setattr("tasks.sae_runtime.loaded_pair_session", lambda pair: nullcontext())
    calls = []

    def measure(child, *, prompt):
        assert prompt == child.task.prompt
        calls.append((child.task.prompt, child.task.feature_id))
        assert child.task.feature_selection == "configured"
        if operation == "generate":
            assert len(child.task.conditions) == 1
            assert list(child.task.conditions[0].feature_ids) == [child.task.feature_id]
        child.task.feature_id = 999  # A child cannot alter the next experiment/config.
        return {"status": "executed", "operation": operation}

    result = run_top_features(cfg, operation, measure)
    assert len(calls) == 4
    assert calls[0][0] == calls[1][0] == "p1"
    assert calls[2][0] == calls[3][0] == "p2"
    assert all(feature != 999 for _, feature in calls)
    assert result["completion"]["completed"] == 4
    assert result["completion"]["status"] == "complete"
    assert len(result["samples"]) == 2
    assert len(open(f"reports/{operation}_features.jsonl").readlines()) == 4
    assert result["required_artifacts"][0]["sha256"]


def test_saved_capture_cannot_cross_numerical_policies(corpus, monkeypatch, tmp_path):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    cfg.task.top_feature_num = 2
    execute(cfg)
    cfg.task.feature_selection = "capture_top"
    cfg.task.feature_ranking = "whiten"
    cfg.task.capture_source = str(tmp_path)
    monkeypatch.setattr("models.numerics.identity", lambda: {"protocol_id": "different-policy"})
    with pytest.raises(ValueError, match="different numerical policy"):
        list(capture_top_cohort(cfg, pair))


def test_saved_whiten_reused_without_refitting_and_rejects_wrong_weights(
    corpus, monkeypatch, tmp_path
):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    cfg.task.top_feature_num = 2
    execute(cfg)
    cfg.task.feature_selection = "capture_top"
    cfg.task.feature_ranking = "whiten"
    cfg.task.capture_source = str(tmp_path)
    monkeypatch.setattr(
        "tasks.capture.whitening.fit_normalization",
        lambda *a: pytest.fail("must reuse saved ranking"),
    )
    selected = list(capture_top_cohort(cfg, pair))
    assert len(selected) == 2 and selected[0]["capture_source"]["ledger_sha256"]
    monkeypatch.setattr(
        "tasks.sae_runtime.pair_state_fingerprints", lambda pair: ("wrong", "b" * 64)
    )
    with pytest.raises(ValueError, match="fingerprints"):
        list(capture_top_cohort(cfg, pair))


@pytest.mark.parametrize(
    "update",
    [
        {"top_feature_num": 0},
        {"top_feature_num": True},
        {"feature_ranking": "sar"},
        {"conditions": [{"operation": "ablate"}]},
        {"protocol": {"conditions": []}},
    ],
)
def test_invalid_top_selection_rejected(update):
    cfg = OmegaConf.create({"dataset": {}, "task": {"feature_selection": "capture_top", **update}})
    with pytest.raises(ValueError):
        validate_top_selection(cfg)


def test_rank_shortage_and_missing_saved_prompt_are_explicit(corpus, tmp_path):
    from tasks.capture.operation import execute

    cfg, pair = corpus
    cfg.task.top_feature_num = 2
    execute(cfg)
    cfg.task.feature_selection = "capture_top"
    cfg.task.feature_ranking = "whiten"
    cfg.task.capture_source = str(tmp_path)
    cfg.task.prompt = "unseen"
    with pytest.raises(ValueError, match="No matching prompt"):
        list(capture_top_cohort(cfg, pair))
    cfg.task.prompt = ""
    cfg.task.top_feature_num = 3
    with pytest.raises(ValueError, match="requires 3 distinct"):
        list(capture_top_cohort(cfg, pair))


def test_completed_feature_row_survives_later_failure(corpus, monkeypatch):
    from tasks.intervene.top_features import run_top_features

    cfg, pair = corpus
    cfg.task.feature_selection = "capture_top"
    cfg.task.top_feature_num = 2
    monkeypatch.setattr("tasks.sae_runtime._pair", lambda cfg: pair)
    calls = []

    def measure(child, *, prompt):
        calls.append(child.task.feature_id)
        if len(calls) == 2:
            raise RuntimeError("synthetic second edit failed")
        return {"status": "executed"}

    with pytest.raises(RuntimeError, match="second edit failed"):
        run_top_features(cfg, "intervene", measure)
    assert len(open("reports/intervene_features.jsonl").readlines()) == 1


def test_resume_reuses_completed_and_retries_only_failures(corpus, monkeypatch):
    from tasks.intervene.top_features import run_top_features

    cfg, pair = corpus
    cfg.task.feature_selection = "capture_top"
    cfg.task.feature_ranking = "non_whiten"
    cfg.task.top_feature_num = 2
    cfg.task.continue_on_error = True
    cfg.task.inline_results = False
    monkeypatch.setattr("tasks.sae_runtime._pair", lambda cfg: pair)
    calls = []

    def fail_one(child, *, prompt):
        calls.append((prompt, child.task.feature_id))
        if len(calls) == 2:
            raise RuntimeError("synthetic failed edit")
        return {"status": "executed"}

    first = run_top_features(cfg, "intervene", fail_one)
    assert first["completion"]["completed"] == 3
    assert first["completion"]["counts"]["failed"] == 1
    cfg.runtime = {"mode": "resume"}
    retried = []

    def succeed(child, *, prompt):
        retried.append((prompt, child.task.feature_id))
        return {"status": "executed"}

    second = run_top_features(cfg, "intervene", succeed)
    assert retried == [calls[1]]
    assert second["reused_measurements"] == 3
    assert second["completion"]["completed"] == 4
    assert len(open("reports/intervene_features.jsonl").readlines()) == 5
    cfg.task.value = 9.0
    with pytest.raises(ValueError, match="outside this cohort/protocol"):
        run_top_features(cfg, "intervene", lambda *a, **kw: pytest.fail("must not execute"))


def test_resume_repairs_only_a_torn_final_line(tmp_path):
    from tasks.intervene.top_features import _load_checkpoint

    path = tmp_path / "rows.jsonl"
    good = json.dumps({"measurement_id": "good", "status": "completed"}) + "\n"
    path.write_text(good + '{"measurement_id":')
    assert list(_load_checkpoint(path)) == ["good"]
    assert path.read_text() == good
    path.write_text(good + "{broken}\n" + good)
    with pytest.raises(ValueError):
        _load_checkpoint(path)
