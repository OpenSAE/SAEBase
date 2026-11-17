"""Synthetic numerical, throughput-search and report contracts; not model benchmarks."""

from types import SimpleNamespace
import json
import numpy as np
import torch
import pytest
from omegaconf import OmegaConf

from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
from saebase.api.replay import SiteReplay
from saebase.runtime.performance import settings, tune, numeric_validation, Recorder
from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE


def options():
    return settings(
        OmegaConf.create(
            {
                "task": {
                    "performance": {
                        "enabled": True,
                        "warmup": 0,
                        "repeats": 1,
                        "batch_sizes": [1, 2, 4, 8],
                        "max_seconds": 30,
                    }
                }
            }
        )
    )


def test_loaded_pair_policy_scope_reuses_searches_between_task_sessions(tmp_path, monkeypatch):
    from saebase.runtime.performance import policy_scope, session, remember_recommendations

    cfg = OmegaConf.create(
        {
            "task": {
                "name": "capture",
                "execute": True,
                "device": "cpu",
                "performance": {"enabled": True},
            }
        }
    )
    monkeypatch.chdir(tmp_path)
    key = ("model", "site", 256, 64, "answer-shape-v2")
    with policy_scope():
        with session(cfg) as first:
            first.answer_policies[key] = 32
        remember_recommendations([])
        with session(cfg) as second:
            assert second.answer_policies[key] == 32
            second.answer_policies[("another",)] = 8
        with session(cfg) as third:
            assert third.answer_policies[("another",)] == 8
    with session(cfg) as isolated:
        assert not getattr(isolated, "answer_policies", {})


def test_answer_policy_buckets_lengths_without_merging_distinct_workloads():
    from tasks.capture.performance import answer_policy_key

    pair = SimpleNamespace(identity=SimpleNamespace(model_key="model"))
    snapshot = SimpleNamespace(tokens=torch.zeros(1, 150), site=SimpleNamespace(hook_name="site"))
    first = answer_policy_key(pair, snapshot, [1] * 13, 64)
    assert first == answer_policy_key(pair, snapshot, [1] * 24, 64)
    assert first != answer_policy_key(pair, snapshot, [1] * 24, 16)
    assert first != answer_policy_key(pair, snapshot, [1] * 200, 64)


def test_answer_policy_remembers_oom_reduction(tmp_path, monkeypatch):
    from tasks.capture import performance as answers
    from saebase.runtime.performance import session

    cfg = OmegaConf.create(
        {
            "task": {
                "name": "capture",
                "execute": True,
                "device": "cpu",
                "performance": {"enabled": True},
            }
        }
    )
    monkeypatch.chdir(tmp_path)
    pair = SimpleNamespace(identity=SimpleNamespace(model_key="model"))
    snapshot = SimpleNamespace(tokens=torch.zeros(1, 10), site=SimpleNamespace(hook_name="site"))
    values = torch.arange(8, dtype=torch.float32)[:, None]
    failures = []

    class Replay:
        def __init__(self, *args):
            self.replay = SimpleNamespace(execution="cached_suffix")

        def __call__(self, activations, batch):
            if len(activations) > 4 and batch > 2:
                failures.append(batch)
                raise torch.cuda.OutOfMemoryError("synthetic out of memory")
            return activations[:, 0].numpy()

    monkeypatch.setattr(answers, "AnswerReplay", Replay)
    key = answers.answer_policy_key(pair, snapshot, [1], 8)
    with session(cfg) as recorder:
        recorder.answer_policies = {key: 4}
        for _ in range(2):
            actual, evidence = answers.evaluate_edits(
                pair,
                snapshot,
                values,
                [1],
                torch.zeros(1),
                lambda value: float(value[0]),
                options(),
            )
            assert np.array_equal(actual, np.arange(8))
            assert evidence["runtime_batch_size"] == 2
        assert recorder.answer_policies[key] == 2
    assert failures == [4]


def test_feature_tuner_compares_warm_preparation_and_records_cold_start(tmp_path, monkeypatch):
    from tasks.intervene import performance as feature_performance
    from saebase.runtime.performance import session

    cfg = OmegaConf.create(
        {
            "task": {
                "name": "intervene",
                "execute": True,
                "device": "cpu",
                "performance": {"enabled": True, "warmup": 1},
            },
        }
    )
    monkeypatch.chdir(tmp_path)
    prepare_calls = []
    pair = SimpleNamespace(identity=SimpleNamespace(model_key="synthetic"))

    def prepare(*args):
        prepare_calls.append(True)
        setup = 100.0 if len(prepare_calls) == 1 else 0.1
        snapshot = SimpleNamespace(tokens=torch.zeros(1, 2))
        values = [{"score": 1.0}, {"score": 2.0}]
        return [None, None], "prompt", snapshot, lambda: values, lambda batch: values, [], setup

    def search(candidate, baseline, units, options, **kwargs):
        assert kwargs["candidate_setup_seconds"] == 0.1
        return {"selected": True, "selected_batch_size": 2, "total_tuning_seconds": 1.0}

    monkeypatch.setattr(feature_performance, "_prepare_features", prepare)
    monkeypatch.setattr(feature_performance, "tune", search)
    with session(cfg):
        output, evidence = feature_performance.measure_features(cfg, "intervene", {}, pair, None)
    assert output == [{"score": 1.0}, {"score": 2.0}]
    assert evidence["selected_batch_size"] == 2
    assert evidence["preparation_warmup_seconds"] == 100.0
    assert evidence["total_tuning_seconds"] == 101.0


def test_tuner_rejects_numerical_changes_and_preserves_baseline():
    expected = np.arange(8, dtype=float)
    report = tune(
        lambda batch: expected + (0 if batch == 1 else 0.01),
        lambda: expected,
        8,
        options(),
        validate=lambda a, b: numeric_validation(a, b, options()),
    )
    assert report["selected_batch_size"] == 1
    assert all(row["status"] == "numerically_rejected" for row in report["trials"][1:])


def test_tuner_records_oom_and_does_not_search_larger_batches():
    def call(batch):
        if batch >= 4:
            raise torch.cuda.OutOfMemoryError("synthetic out of memory")
        return np.arange(8, dtype=float)

    report = tune(
        call,
        lambda: np.arange(8, dtype=float),
        8,
        options(),
        validate=lambda a, b: numeric_validation(a, b, options()),
    )
    assert report["trials"][-1]["status"] == "out_of_memory"
    assert [row["batch_size"] for row in report["trials"]] == [1, 2, 4]


@pytest.mark.parametrize("live_peak", [400, 950])
def test_tuner_distinguishes_historical_cache_from_actual_memory_pressure(monkeypatch, live_peak):
    from saebase.runtime import performance

    # A previous input reserved 95% of the device. Only 10% remains live.
    state = {"reserved": 950, "duration": 0.1, "cleanups": 0}

    def cleanup():
        state["reserved"] = 100
        state["cleanups"] += 1

    def baseline():
        state.update(reserved=950, duration=0.1)
        return np.arange(8, dtype=float)

    def candidate(batch):
        state["reserved"] = max(state["reserved"], live_peak)
        state["duration"] = 0.01 / batch
        return np.arange(8, dtype=float)

    def measure(call):
        value = call()
        return value, state["duration"]

    monkeypatch.setattr(performance, "cuda_measurement", lambda: True)
    monkeypatch.setattr(performance, "timed", measure)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: state["reserved"])
    monkeypatch.setattr(torch.cuda, "empty_cache", cleanup)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: live_peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: state["reserved"])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=1000)
    )
    report = tune(
        candidate, baseline, 8, options(), validate=lambda a, b: numeric_validation(a, b, options())
    )
    assert state["cleanups"] == len(report["trials"])
    assert report["trials"][0]["reserved_before_cleanup_bytes"] == 950
    if live_peak == 400:
        assert report["selected"] and report["selected_batch_size"] == 8
        assert all(row["status"] == "accepted" for row in report["trials"])
    else:
        assert not report["selected"]
        assert all(row["status"] == "memory_headroom_rejected" for row in report["trials"])


def test_cached_hf_batch_matches_independent_full_forwards():
    from tests.product.test_suffix_replay import wrapper

    raw = wrapper()
    model = ModelAdapter(raw)
    ids = torch.tensor([[1, 2, 3]])
    site = ActivationSite(0, "blocks.0.hook_resid_post", -1)
    original = model.capture(tokens=ids, site=site)
    targets = torch.stack([original, original + 0.2, original - 0.1])
    replay = SiteReplay(model, ids, site)
    assert replay.execution == "cached_suffix"
    actual = replay.logits(targets)
    expected = torch.cat(
        [raw.run_with_patch(ids, 0, {0: target}, readout_offset=None) for target in targets]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(raw(ids), raw(ids))


def test_native_decoder_reuse_and_dedup_are_exact():
    from tasks.capture.performance import prepare_activations

    raw = ResearchToyModel()
    model = ModelAdapter(raw)
    sae = ResearchToySAE()
    snapshot = capture_features(model, sae, site=ActivationSite(0, raw.site_name), prompt="0 1 0")
    ids = [0, 1, 2]  # Feature 0 and 2 are zero at the final token.
    masks = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)
    activations, inverse, info = prepare_activations(sae, snapshot, ids, masks)
    for index, mask in enumerate(masks):
        reference = edit_features(
            sae, snapshot, replacements={fid: 0.0 for fid, bit in zip(ids, mask) if not bit}
        ).activation
        torch.testing.assert_close(activations[inverse[index]], reference, atol=0, rtol=0)
    assert info["unique_native_code_states"] == 2


def test_vectorized_pair_ranking_matches_brute_force():
    from tasks.capture.behavior import choose_pair
    from itertools import combinations

    rng = np.random.default_rng(7)
    values = -rng.uniform(0, 5, 32)
    actual = choose_pair(list(range(5)), values, 2, return_ranking=True)
    for row in actual:
        i, j = row["pair_ids"]
        best = None
        for background in range(32):
            if background & ((1 << i) | (1 << j)):
                continue
            value = (
                values[background | (1 << i) | (1 << j)]
                - values[background | (1 << i)]
                - values[background | (1 << j)]
                + values[background]
            )
            if best is None or value > best[0]:
                best = (value, background)
        assert row["interaction"] == best[0] and row["states"]["00"] == best[1]
    assert len(actual) == len(list(combinations(range(5), 2)))


def test_native_generation_batch_matches_scalar_tokens_and_step_metrics():
    from tasks.generate.backend.native_batch import compare_many
    from tasks.intervene.backend.steering import GenerationSpec, compare_steered_generation
    from tasks.intervene.backend.specs import Condition, MeasurementSample
    from saebase.store import SampleRecord, TokenSelection
    from tasks.intervene.performance import validate_outputs

    raw = ResearchToyModel()
    model = ModelAdapter(raw)
    sae = ResearchToySAE()
    pair = SimpleNamespace(model=model, sae=sae, site=ActivationSite(0, raw.site_name))
    record = SampleRecord(
        sample_id="synthetic", token_ids=(1, 2), selection=TokenSelection("last_valid")
    )
    sample = MeasurementSample(record)
    spec = GenerationSpec(max_new_tokens=3, seed=42)
    conditions = [Condition("delete", "ablate", (0,)), Condition("scale", "scale", (1,), 2.0)]
    expected = [
        compare_steered_generation(
            model, sae, site=pair.site, sample=sample, condition=c, spec=spec
        )
        for c in conditions
    ]
    for batch in [1, 2]:
        actual = compare_many(pair, sample, conditions, spec, batch)
        assert validate_outputs(expected, actual, options())["accepted"]


def test_report_contains_distributions_phases_and_raw_observations(tmp_path):
    recorder = Recorder("synthetic", tmp_path)
    recorder.sample("a", 1.0)
    recorder.sample("b", 3.0)
    recorder.phases["forward"] = [0.2, 0.3]
    recorder.save()
    report = json.loads((tmp_path / "reports/performance.json").read_text())
    assert report["per_sample"]["question"]["mean_seconds"] == 2.0
    assert report["per_sample"]["question"]["p50_seconds"] == 2.0
    assert report["phases"]["forward"]["total_seconds"] == 0.5
    assert len(report["observations"]) == 2
    assert (tmp_path / "reports/performance.md").is_file()


def test_cached_prompt_observations_match_native_patch():
    from tests.product.test_suffix_replay import wrapper
    from tasks.intervene.backend.observations import PromptObservations

    raw = wrapper()
    model = ModelAdapter(raw)
    sae = ResearchToySAE()
    tokens = torch.tensor([[1, 2, 3]])
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    snapshot = capture_features(model, sae, site=site, tokens=tokens)
    pair = SimpleNamespace(model=model, sae=sae, site=site)
    observations = PromptObservations(pair, suffix_cache=True)
    for value in [0.1, 0.3, -0.2]:
        activation = snapshot.activation + value
        actual = observations.patch(snapshot, activation)
        expected = model.patch_logits(
            tokens=tokens, site=site, activation=activation, readout_position=-1
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(observations.replays) == 1


def test_answer_replay_moves_cpu_saved_snapshots_to_the_model_device():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the mixed-device regression")
    from dataclasses import fields, replace
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from tasks.capture.performance import AnswerReplay

    torch.manual_seed(5)
    raw = HookedTransformer(
        HookedTransformerConfig(
            n_layers=2,
            d_model=4,
            d_head=4,
            n_heads=1,
            d_mlp=8,
            n_ctx=16,
            d_vocab=7,
            act_fn="relu",
            device="cuda",
        )
    ).eval()
    model = ModelAdapter(raw)
    sae = ResearchToySAE().cuda()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    prompt = torch.tensor([[1, 2, 3]], device="cuda")
    answer = torch.tensor([[4, 5]], device="cuda")
    original = capture_features(model, sae, site=site, tokens=prompt)
    snapshot = replace(
        original,
        **{
            f.name: getattr(original, f.name).cpu()
            for f in fields(original)
            if isinstance(getattr(original, f.name), torch.Tensor)
        },
    )
    joint = torch.cat([prompt, answer[:, :-1]], 1)
    reference = model.capture(tokens=joint, site=ActivationSite(0, site.hook_name, 2))
    pair = SimpleNamespace(model=model, sae=sae)
    scorer = AnswerReplay(pair, snapshot, [4, 5], reference)
    activations = torch.stack([snapshot.activation, snapshot.activation + 0.3])
    actual = scorer(activations, 1)
    expected = []
    for activation in activations:
        target = reference + (activation.cuda() - original.activation)
        expected.append(
            float(
                model.answer_token_logprobs(
                    prompt_tokens=prompt, answer_tokens=answer, site=site, activation=target
                )
                .double()
                .mean()
            )
        )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


def test_performance_configuration_does_not_load_weights(monkeypatch):
    from tasks.precheck.task import PrecheckTask

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration must not load weights")

    monkeypatch.setattr("tasks.sae_runtime.load_selection", forbidden)
    cfg = OmegaConf.create(
        {
            "model": {"name": "gpt2small"},
            "dataset": {"name": "GSM8K"},
            "task": {
                "name": "precheck",
                "mode": "performance",
                "execute": False,
                "performance": {"enabled": True},
                "check_tasks": ["capture"],
                "sae_list": "default",
            },
        }
    )
    result = PrecheckTask().run(cfg)
    assert result.status == "configured_only" and result.summary["status"] == "NOT_RUN"


def test_setup_cost_prevents_selecting_a_slower_complete_probe():
    expected = np.arange(8, dtype=float)
    result = tune(
        lambda batch: expected,
        lambda: expected,
        8,
        options(),
        validate=lambda a, b: numeric_validation(a, b, options()),
        candidate_setup_seconds=1,
    )
    assert not result["selected"] and result["speedup"] == 1
    assert result["candidate_setup_seconds"] == 1


def test_runtime_preserves_task_owned_performance_report_paths(tmp_path, monkeypatch):
    from saebase.registry import TaskModule
    from saebase.results import EvaluationResult
    from saebase.runner import run_task

    class Checked(TaskModule):
        name = "precheck"

        def run(self, cfg):
            return EvaluationResult(
                ok=True,
                task="precheck",
                status="executed",
                evaluation="synthetic",
                model_key="synthetic",
                prompt_source="synthetic",
                artifact_paths={
                    "performance": "task-owned.json",
                    "performance_report": "task-owned.md",
                },
            )

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create(
        {
            "model": {"name": "synthetic", "device": "cpu"},
            "dataset": {"name": "synthetic"},
            "task": {
                "name": "precheck",
                "execute": True,
                "device": "cpu",
                "performance": {"enabled": True},
            },
        }
    )
    result = run_task(cfg, task=Checked())
    assert result.artifact_paths["performance"] == "task-owned.json"
    assert result.artifact_paths["performance_report"] == "task-owned.md"
    assert (tmp_path / "reports/performance.json").is_file()


def test_recommendations_transfer_once_and_keep_native_fallback(tmp_path, monkeypatch):
    from saebase.runtime.performance import remember_recommendations, session

    monkeypatch.chdir(tmp_path)
    feature = ["generate", "synthetic", 1, 10, 8]
    answer = ["synthetic", "blocks.0.hook_resid_post", 1, 2]
    remember_recommendations(
        [
            {
                "optimizations": [
                    {
                        "selected": True,
                        "policy_kind": "feature",
                        "policy_key": feature,
                        "selected_batch_size": 4,
                    },
                    {
                        "selected": False,
                        "policy_kind": "answer",
                        "policy_key": answer,
                        "selected_batch_size": 1,
                    },
                ]
            }
        ]
    )
    cfg = OmegaConf.create(
        {
            "task": {
                "name": "synthetic",
                "execute": True,
                "device": "cpu",
                "performance": {"enabled": True},
            }
        }
    )
    with session(cfg) as recorder:
        assert recorder.policies[tuple(feature)] == 4
        assert recorder.answer_policies[tuple(answer)] == 0
    with session(cfg) as recorder:
        assert not getattr(recorder, "policies", {})


def test_generation_rank_shortcut_preserves_reference_tie_order():
    from tasks.intervene.backend.steering import _distribution_shift

    clean = torch.tensor([3.0, 1.0, 0.0, -1.0])
    for edited in [torch.tensor([0.5, 2.0, 0.0, -1.0]), torch.tensor([0.0, 2.0, 2.0, 0.0])]:
        expected = int((torch.argsort(edited, descending=True) == 0).nonzero()[0].item())
        actual = _distribution_shift(clean, edited, feature_id=0, activation=1.0)
        assert actual["clean_argmax_rank_after_edit"] == expected
