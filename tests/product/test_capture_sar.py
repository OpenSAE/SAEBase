"""Synthetic SAR mounting tests; the estimator double is not a scientific result."""

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from saebase.api import ModelAdapter, SAEIdentity, capture_features, edit_features
from tasks.capture import operation
from tasks.capture.sar_mode import behavior, prepare_sar
from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE


def test_sar_filter_matches_forms_without_removing_topical_or_code_labels():
    from tasks.capture.sar_filter import blocked_features

    labels = [
        "punctuation marks",
        "the word 'on' indicating ongoing actions",
        "rhetorical questions",
        "Python syntax and programming constructs",
        "colon cancer",
        "newspaper articles",
        "the word 'May' as a calendar month",
        "the word 'Will' as a person name",
        "references to New Mexico",
        "broad content",
    ]
    descriptions = {i: {"explanations": [{"description": text}]} for i, text in enumerate(labels)}
    densities = {str(i): 0.1 for i in descriptions}
    densities["9"] = 0.5
    blocked = blocked_features(descriptions, densities, 0.5)
    assert set(blocked) == {0, 1, 2, 9}
    assert blocked[9]["rules"] == ["widespread_native_activation"]


def test_cohort_filter_counts_signed_exceedances_over_all_questions(tmp_path, monkeypatch):
    from tasks.capture.sar_filter import prepare_cohort_filter, select_features
    from tasks.capture.whitening import Normalization

    monkeypatch.chdir(tmp_path)
    codes = torch.tensor(
        [[1.0, 1.1 if i < 5 else 0.0, 1.1 if i < 4 else 0.0, -2.0] for i in range(10)]
    )
    rows = [{"sample_id": str(i), "prompt": str(i)} for i in range(10)]
    snapshots = [SimpleNamespace(codes=c) for c in codes]
    norm = Normalization(np.zeros(4), np.ones(4), np.ones(4), {"source": "synthetic"}, {})
    context = {
        "settings": {
            "cohort_filter": {
                "enabled": True,
                "zscore_threshold": 1.0,
                "min_question_fraction": 0.5,
            },
            "feature_filter": {"enabled": False},
        }
    }
    prepare_cohort_filter(context, norm, snapshots, rows)
    assert context["cohort_blocked_features"] == {1}
    saved = json.loads(Path("metadata/sar_cohort_filter.json").read_text())
    assert saved["exceedance_counts"] == [0, 5, 4, 0]
    # z == 1 stays eligible, four out of ten stays eligible, negative z is not counted.
    assert select_features(None, codes[0], norm, 3, context, {}) == [3, 2, 0]
    assert context["cohort_filter_identity"]["question_count"] == 10


def test_cohort_filter_requires_full_cohort_before_sar_selection(mounted):
    cfg, pair = mounted
    cfg.task.sar.feature_selection = "whiten"
    cfg.task.sar.cohort_filter = {"enabled": True}
    context = prepare_sar(cfg)
    from tasks.capture.sar_filter import select_features

    with pytest.raises(ValueError, match="all selected questions"):
        select_features(pair, torch.zeros(7), None, 3, context, {})


def test_signed_pair_ranking_uses_saved_coefficients_without_absolute_value(mounted):
    cfg, _ = mounted
    cfg.task.sar.top_pairs = 3
    cfg.task.sar.pair_ranking = "signed"
    result = operation.execute(cfg)["sar"]
    assert result["pairwise"]["ranking"] == "signed_pairwise_coefficient_descending"
    with np.load(result["artifact"]["path"]) as saved:
        coefficients = saved["pair_coefficients"]
    reported = [p["coefficient"] for p in result["pairwise"]["top_pairs"]]
    assert reported == sorted(coefficients.tolist(), reverse=True)


def test_reference_100_questions_only_measures_ten_and_matches_snapshots_by_identity(
    mounted, monkeypatch
):
    from saebase.datasets.selection import SampleSelection

    cfg, pair = mounted
    cfg.task.prompt = ""
    cfg.dataset.sample_count = 10
    cfg.task.sar.feature_selection = "whiten"
    cfg.task.sar.cohort_filter = {
        "enabled": True,
        "reference_sample_count": 100,
        "zscore_threshold": 1000.0,
        "min_question_fraction": 0.5,
    }
    reference = [{"sample_id": str(i), "prompt": f"{2 + i % 3} 3 4"} for i in range(100)]
    target = [reference[i] for i in [97, 3, 22, 51, 8, 41, 90, 63, 12, 35]]
    requested = []

    def select(config, **kwargs):
        requested.append(config.sample_count)
        rows = reference if config.sample_count == 100 else target
        return SampleSelection(rows, {"selected_count": len(rows), "source": "synthetic"})

    monkeypatch.setattr("saebase.datasets.selection.select_dataset", select)
    capture = operation._capture
    measured = []

    def checked(config, **kwargs):
        measured.append(kwargs["sample"]["sample_id"])
        assert (
            kwargs["snapshot"].tokens.tolist()
            == pair.model.tokens(prompt=kwargs["prompt"]).tolist()
        )
        return capture(config, **kwargs)

    monkeypatch.setattr(operation, "_capture", checked)
    result = operation.execute(cfg)
    assert requested == [10, 100]
    assert measured == [r["sample_id"] for r in target]
    assert result["completion"]["requested"] == result["completion"]["completed"] == 10
    assert (
        json.loads(Path("metadata/capture_normalization.json").read_text())["prompt_count"] == 100
    )
    assert (
        json.loads(Path("metadata/sar_cohort_filter.json").read_text())["identity"][
            "question_count"
        ]
        == 100
    )


def test_sar_filter_precedes_top_k_and_does_not_change_other_capture_modes(mounted, monkeypatch):
    cfg, pair = mounted
    snapshot = capture_features(pair.model, pair.sae, prompt=cfg.task.prompt, site=pair.site)
    ranked = torch.argsort(snapshot.codes.abs(), descending=True, stable=True).tolist()
    excluded = ranked[0]
    calls = []

    def annotate(pair, ids, options):
        if not options["enabled"]:
            return {"status": "disabled"}, {
                i: {"status": "disabled", "explanations": []} for i in ids
            }
        calls.append(ids)
        return {"status": "complete", "neuronpedia_id": "synthetic/0", "export_sha256": "test"}, {
            i: {"explanations": [{"description": "punctuation" if i == excluded else "New Mexico"}]}
            for i in ids
        }

    monkeypatch.setattr("saebase.api.neuronpedia.annotate_features", annotate)
    monkeypatch.setattr(
        "saebase.api.neuronpedia.feature_activation_densities",
        lambda *args: {
            "fractions": {str(i): 0.1 for i in ranked},
            "sha256": "test",
            "provenance": {},
        },
    )
    cfg.task.sar.feature_filter = {"enabled": True}
    context = prepare_sar(cfg)
    from tasks.capture.sar_filter import select_features
    from tasks.capture.whitening import Normalization

    # The same filter is applied to complete native and whiten rankings.
    normalization = Normalization(np.zeros(7), np.ones(7), np.ones(7), {}, {})
    for norm in (None, normalization):
        selected = select_features(pair, snapshot.codes, norm, 3, context, {})
        assert selected == ranked[1:4]
    assert len(calls) == 1  # One frozen blocklist for the run.
    result = operation.execute(cfg)
    assert result["sar"]["feature_ids"] == ranked[1:4]
    assert result["sar"]["identity"]["feature_filter"]["blocked_count"] == 1
    assert Path("metadata/sar_feature_filter.json").is_file()

    # Annotation is disabled for these normal captures; filtering must not load metadata.
    calls.clear()
    cfg.task.mode = "non_whiten"
    unfiltered = operation.execute(cfg)
    assert excluded in [f["feature_id"] for f in unfiltered["top_features"]]
    cfg.task.mode = "whiten"
    whitened = operation._capture(cfg, pair=pair, normalization=normalization)
    assert excluded in [f["feature_id"] for f in whitened["top_features"]]
    assert calls == []


def test_sar_density_exports_are_source_bound_and_reusable_offline(tmp_path, monkeypatch):
    import gzip

    import saebase.api.neuronpedia as api

    expected = {
        "id": "synthetic/0",
        "repo": "test",
        "folder": "folder",
        "d_in": 4,
        "d_sae": 2,
        "hook": "blocks.0.hook_resid_post",
    }
    source = {
        "modelId": "synthetic",
        "id": "0",
        "hfRepoId": "test",
        "hfFolderId": "folder",
        "saelensConfig": {"d_in": 4, "d_sae": 2, "hook_name": expected["hook"]},
    }
    monkeypatch.setattr(api, "_source", lambda pair: expected)
    rows = [
        {"modelId": "synthetic", "layer": "0", "index": str(i), "frac_nonzero": v}
        for i, v in enumerate([0.5, None])
    ]
    packed = gzip.compress("\n".join(json.dumps(r) for r in rows).encode())
    listing = (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<IsTruncated>false</IsTruncated><Contents>"
        "<Key>v1/synthetic/0/features/batch-0.jsonl.gz</Key>"
        f"<Size>{len(packed)}</Size></Contents></ListBucketResult>"
    ).encode()

    def fetch(url, **kwargs):
        if url.endswith("source.jsonl"):
            return json.dumps(source).encode()
        return packed if url.endswith(".gz") else listing

    monkeypatch.setattr(api, "_fetch", fetch)
    options = {
        "cache_dir": str(tmp_path),
        "allow_network": True,
        "total_timeout_seconds": 10,
        "timeout_seconds": 10,
        "max_download_bytes": 10000,
        "max_uncompressed_bytes": 10000,
    }
    result = api.feature_activation_densities(None, options)
    assert result["fractions"] == {"0": 0.5, "1": None}
    monkeypatch.setattr(api, "_fetch", lambda *a, **k: pytest.fail("cached read used network"))
    assert api.feature_activation_densities(None, {**options, "allow_network": False}) == result
    expected["d_sae"] = 3
    with pytest.raises(ValueError):
        api.feature_activation_densities(None, options)


# Deliberately fixed SAR values test data plumbing, not the SAR estimator math.
PAPER_DOUBLE = '''"""Synthetic estimator test double, not the original SAR implementation."""
import numpy as np

def coalitions(K, M, p, n_fit, seed):
    rng = np.random.default_rng(seed)
    masks = rng.random((M, K)) < p
    order = rng.permutation(M)
    return masks, order[:n_fit], order[n_fit:]

def sar(g, masks, fit, held, g_full):
    assert len(g) == len(masks)
    assert not set(fit).intersection(held)
    return {"sar1": 0.25, "sar2": 0.1, "var": float(np.var(g[held])),
            "rho2": float("nan"), "baseline": float(g_full)}

def design(masks, order):
    from itertools import combinations
    assert order == 2
    s = masks.astype(float)
    return np.column_stack([np.ones(len(s)), s] + [s[:, i]*s[:, j] for i,j in combinations(range(s.shape[1]),2)])
'''


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    import tasks.sae_runtime as runtime

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sar.py"
    source.write_text(PAPER_DOUBLE)
    raw = ResearchToyModel().eval()
    pair = SimpleNamespace(
        model=ModelAdapter(raw),
        sae=ResearchToySAE().eval(),
        identity=SAEIdentity(
            model_key="synthetic",
            sae_key="synthetic",
            checkpoint="synthetic",
            sae_id="synthetic",
            hook_name=raw.site_name,
            layer=0,
            loader="synthetic",
            d_in=4,
            d_sae=7,
        ),
    )
    pair.site = pair.identity.site
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    monkeypatch.setattr(runtime, "loaded_pair_session", lambda pair: nullcontext())
    monkeypatch.setattr(runtime, "pair_state_fingerprints", lambda pair: ("a" * 64, "b" * 64))
    cfg = OmegaConf.create(
        {
            "seed": 3,
            "model": {"name": "synthetic"},
            "dataset": {"name": "synthetic", "sample_count": 2},
            "task": {
                "name": "capture",
                "mode": "sar",
                "top_feature_num": 3,
                "prompt": "2 3 4",
                "neuronpedia": {"enabled": False},
                "sar": {
                    "source": str(source),
                    "M": 80,
                    "n_fit": 55,
                    "target_token_id": 0,
                    "reference_token_id": 1,
                },
            },
        }
    )
    return cfg, pair


def test_single_capture_reuses_snapshot_and_saves_exact_coalition_measurements(
    mounted, tmp_path, monkeypatch
):
    cfg, pair = mounted
    calls = []
    original = pair.model.model.run_with_cache

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pair.model.model, "run_with_cache", capture)
    result = operation.execute(cfg)
    assert len(calls) == 1
    assert result["completion"]["status"] == "complete"
    sar = result["sar"]
    assert sar["metrics"]["sar1"] == 0.25 and sar["metrics"]["rho2"] is None
    assert sar["undefined_metrics"] == ["rho2"]
    assert sar["feature_ids"] == [f["feature_id"] for f in result["top_features"]]
    assert result["input_identity"]["sar"] == sar["identity"]
    artifact = tmp_path / sar["artifact"]["path"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == sar["artifact"]["sha256"]
    snapshot = capture_features(pair.model, pair.sae, prompt=cfg.task.prompt, site=pair.site)
    with np.load(artifact) as saved:
        assert len(saved["masks"]) == 80
        assert len(saved["fit_indices"]) == 55 and len(saved["held_indices"]) == 25
        expected = []
        for mask in saved["masks"]:
            replacements = {fid: 0.0 for fid, keep in zip(sar["feature_ids"], mask) if not keep}
            patch = edit_features(pair.sae, snapshot, replacements=replacements)
            logits = pair.model.patch_logits(
                tokens=snapshot.tokens, site=pair.site, activation=patch.activation
            )
            expected.append(float(logits[0] - logits[1]))
        np.testing.assert_allclose(saved["responses"], expected, atol=1e-7)
        np.testing.assert_array_equal(saved["source_tokens"], snapshot.tokens.numpy())
    assert result["required_artifacts"] == [sar["artifact"]]


def test_dataset_uses_each_records_labels_and_registers_raw_artifacts(
    mounted, tmp_path, monkeypatch
):
    import saebase.datasets.selection as selection

    cfg, _ = mounted
    cfg.task.prompt = ""
    cfg.task.sar.target_token_id = None
    cfg.task.sar.reference_token_id = None
    rows = [
        {
            "sample_id": "a",
            "prompt": "2 3 4",
            "tokenization": {"correct_ids": [0], "distractor_ids": [1]},
        },
        {"sample_id": "b", "prompt": "2 3 4", "target_token_id": 1, "reference_token_id": 0},
    ]
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(rows, {"selected_count": 2}),
    )
    result = operation.execute(cfg)
    assert result["completion"]["status"] == "complete" and result["sample_count"] == 2
    ledger = [
        json.loads(line)
        for line in (tmp_path / "reports/capture_samples.jsonl").read_text().splitlines()
    ]
    assert [r["result"]["sar"]["identity"]["behavior"]["target_token_id"] for r in ledger] == [0, 1]
    artifacts = [r["result"]["sar"]["artifact"]["path"] for r in ledger]
    with np.load(tmp_path / artifacts[0]) as a, np.load(tmp_path / artifacts[1]) as b:
        np.testing.assert_allclose(a["responses"], -b["responses"])
    assert all((tmp_path / a).is_file() for a in artifacts)
    assert len(result["required_artifacts"]) == 3
    assert (
        result["input_identity"]["sar"]["source_sha256"]
        == hashlib.sha256(Path(cfg.task.sar.source).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("M", True),
        ("n_fit", 4),
        ("p", 1),
        ("p", float("nan")),
        ("readout", "guess"),
        ("readout_position", False),
        ("require_affine_decode", "false"),
        ("source", None),
    ],
)
def test_invalid_protocol_rejected_before_model_loading(mounted, monkeypatch, key, value):
    cfg, _ = mounted
    cfg.task.sar[key] = value
    monkeypatch.setattr(operation, "_pair", lambda cfg: pytest.fail("must not load"))
    with pytest.raises(ValueError):
        operation.execute(cfg)


def test_validation_never_executes_paper_source(mounted):
    cfg, _ = mounted
    Path(cfg.task.sar.source).write_text("raise RuntimeError('must not import')")
    operation.validate_cfg(cfg)


@pytest.mark.parametrize(
    "sample",
    [
        {},
        {"target_token_id": 0},
        {"target_token_id": 0, "reference_token_id": 0},
        {"target_token_id": True, "reference_token_id": 1},
        {"tokenization": {"correct_ids": [0, 1], "distractor_ids": [1]}},
    ],
)
def test_missing_or_invalid_labels_are_not_guessed(mounted, sample):
    cfg, _ = mounted
    cfg.task.sar.target_token_id = cfg.task.sar.reference_token_id = None
    with pytest.raises(ValueError):
        behavior(cfg, sample)


def test_conflicting_labels_rejected(mounted):
    cfg, _ = mounted
    with pytest.raises(ValueError, match="Conflicting"):
        behavior(cfg, {"target_token_id": 2})


def test_source_change_and_dictionary_overflow_are_rejected(mounted):
    cfg, _ = mounted
    context = prepare_sar(cfg)
    Path(cfg.task.sar.source).write_text(PAPER_DOUBLE + "\n# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        operation._capture(cfg, sar_context=context)
    cfg.task.top_feature_num = 8
    with pytest.raises(ValueError, match="exceeds SAE feature"):
        operation.execute(cfg)


def test_nonadditive_decode_is_rejected_and_alternative_is_named(mounted, monkeypatch):
    cfg, pair = mounted
    monkeypatch.setattr(
        pair.sae, "decode", lambda z: z[..., :4] + z.sum(dim=-1, keepdim=True).square()
    )
    with pytest.raises(ValueError, match="not additive"):
        operation.execute(cfg)
    cfg.task.sar.require_affine_decode = False
    result = operation.execute(cfg)
    assert "decoder_mediated" in result["sar"]["identity"]["protocol_id"]


def test_old_parameter_is_rejected_instead_of_silently_ignored(mounted):
    cfg, _ = mounted
    cfg.task.top_k = 2
    with pytest.raises(ValueError, match="renamed"):
        operation.execute(cfg)


def test_current_hydra_config_exposes_sar_and_new_count():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=["task=capture", "task.mode=sar", "task.top_feature_num=16"],
        )
    assert "top_k" not in cfg.task and cfg.task.top_feature_num == 16
    assert cfg.task.sar.M == 384 and cfg.task.sar.n_fit == 256


def test_reported_pair_recovers_known_synthetic_model_interaction(mounted):
    from tasks.capture.sar import evaluate_sar

    cfg, pair = mounted
    snapshot = capture_features(pair.model, pair.sae, prompt=cfg.task.prompt, site=pair.site)
    result = evaluate_sar(
        pair.model,
        pair.sae,
        site=pair.site,
        paper=prepare_sar(cfg)["paper"],
        readout=lambda logits: logits[0] - logits[1],
        tokens=snapshot.tokens,
        snapshot=snapshot,
        K=2,
        M=80,
        n_fit=55,
        feature_ids=[0, 1],
        top_pairs=1,
    )
    reported = result.pairwise["top_pairs"][0]
    assert reported["feature_ids"] == [0, 1]
    # Toy logits have 1.4*h0*h1; each retained code contributes 0.8*z.
    expected = 1.4 * 0.8**2 * float(snapshot.codes[0]) * float(snapshot.codes[1])
    assert reported["coefficient"] == pytest.approx(expected, abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one small CUDA decode")
def test_offloaded_whiten_snapshot_decodes_with_gpu_sae(mounted):
    cfg, pair = mounted
    snapshot = capture_features(pair.model, pair.sae, prompt=cfg.task.prompt, site=pair.site)
    expected = edit_features(pair.sae, snapshot, replacements={0: 0.0, 1: 0.0})
    pair.sae.to("cuda")
    actual = edit_features(pair.sae, snapshot, replacements={0: 0.0, 1: 0.0})
    assert actual.activation.device.type == "cpu"
    torch.testing.assert_close(actual.activation, expected.activation, atol=1e-6, rtol=1e-6)


def test_whiten_sar_measures_the_ranked_ids_and_saves_pair_evidence(mounted, monkeypatch):
    from saebase.datasets.selection import SampleSelection

    from tasks.capture.whitening import fit_normalization

    cfg, pair = mounted
    cfg.task.sar.feature_selection = "whiten"
    cfg.task.sar.top_pairs = 3
    row = {
        "sample_id": "known",
        "prompt": cfg.task.prompt,
        "target_token_id": 0,
        "reference_token_id": 1,
    }
    cfg.task.prompt = ""
    monkeypatch.setattr(
        "saebase.datasets.selection.select_dataset",
        lambda *a, **k: SampleSelection([row], {"selected_count": 1}),
    )
    snapshot = capture_features(pair.model, pair.sae, prompt=row["prompt"], site=pair.site)
    fitted = []

    def tracked(*args):
        normalization, snapshots = fit_normalization(*args)
        fitted.append(normalization)
        return normalization, snapshots

    monkeypatch.setattr("tasks.capture.whitening.fit_normalization", tracked)
    result = operation.execute(cfg)
    saved = json.loads(Path(result["samples_path"]).read_text())["result"]
    _, expected_ids = fitted[0].rank(snapshot.codes, 3)
    assert saved["sar"]["feature_ids"] == expected_ids
    assert saved["ranking"] == "absolute_zscore"
    assert all("zscore" in f for f in saved["top_features"])
    assert len(saved["sar"]["pairwise"]["top_pairs"]) == 3
    assert all("features" in p for p in saved["sar"]["pairwise"]["top_pairs"])
    with np.load(saved["sar"]["artifact"]["path"]) as arrays:
        assert arrays["pair_feature_ids"].shape == (3, 2)
        assert arrays["pair_coefficients"].shape == (3,)
    from tasks.capture.report import export_report

    monkeypatch.setattr(
        "saebase.datasets.selection.read_selection",
        lambda *a: SampleSelection([row], {"selected_count": 1}),
    )
    metadata = Path("metadata")
    (metadata / "manifest.json").write_text(
        json.dumps({"configuration": OmegaConf.to_container(cfg), "result": {"summary": result}})
    )
    report = export_report(Path.cwd())
    payload = json.loads(Path(report["json"]).read_text())
    assert payload["recorded"] == 1 and payload["layer"] == 0
    assert payload["questions"][0]["features"] == saved["top_features"]
    assert len(payload["questions"][0]["pairs"]) == 3
    assert "Neuronpedia" in Path(report["html"]).read_text()


def test_original_estimator_matches_direct_call_when_supplied(mounted):
    source = os.environ.get("SAEBASE_SAR_SOURCE")
    if not source:
        pytest.skip("Original SAR source not supplied; synthetic mounting tests run separately")
    cfg, _ = mounted
    cfg.task.sar.source = source
    result = operation.execute(cfg)
    context = prepare_sar(cfg)
    with np.load(result["sar"]["artifact"]["path"]) as saved:
        direct = context["paper"].sar(
            saved["responses"],
            saved["masks"],
            saved["fit_indices"],
            saved["held_indices"],
            g_full=float(saved["full_response"]),
        )
    for key, value in direct.items():
        assert result["sar"]["metrics"][key] == (
            pytest.approx(value) if np.isfinite(value) else None
        )


def test_missing_dataset_label_fails_only_its_sample(mounted, monkeypatch, tmp_path):
    import saebase.datasets.selection as selection

    cfg, _ = mounted
    cfg.task.prompt = ""
    cfg.task.sar.target_token_id = cfg.task.sar.reference_token_id = None
    rows = [
        {"sample_id": "bad", "prompt": "1 2 3"},
        {"sample_id": "good", "prompt": "2 3 4", "target_token_id": 0, "reference_token_id": 1},
    ]
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(rows, {"selected_count": 2}),
    )
    result = operation.execute(cfg)
    assert result["counts"]["failed"] == 1 and result["counts"]["completed"] == 1
    assert result["selection_exhausted"] and result["completion"]["missing"] == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "reports/capture_samples.jsonl").read_text().splitlines()
    ]
    assert "target_token_id" in rows[0]["error"]["message"] and rows[1]["status"] == "completed"


@pytest.mark.parametrize("selected_available", [True, False])
def test_only_selected_explanations_determine_capture_completion(
    mounted, monkeypatch, tmp_path, selected_available
):
    import saebase.datasets.selection as selection

    import saebase.api.neuronpedia as neuronpedia
    from saebase.results import EvaluationResult
    from saebase.runtime.experiment_results import read_run, record_run
    from tasks.capture.annotations import read_samples, refresh_annotations

    cfg, _ = mounted
    cfg.task.mode = "non_whiten"
    cfg.task.top_feature_num = 1
    observed = operation.execute(cfg)
    selected_id = observed["top_features"][0]["feature_id"]
    assert len(observed["active_features"]) > 1
    cfg.task.neuronpedia.enabled = True

    def annotations(pair, ids, options):
        descriptions = {
            i: {
                "status": "available" if i == selected_id and selected_available else "missing",
                "explanations": [{"description": "Synthetic selected feature"}]
                if i == selected_id and selected_available
                else [],
            }
            for i in set(ids)
        }
        return {"status": "partial"}, descriptions

    monkeypatch.setattr(neuronpedia, "annotate_features", annotations)
    monkeypatch.setattr("tasks.capture.annotations.annotate_features", annotations)
    cfg.task.prompt = ""
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(
            [{"sample_id": "selected", "prompt": "2 3 4"}], {"selected_count": 1}
        ),
    )
    raw = operation.execute(cfg)
    expected = "complete" if selected_available else "incomplete"
    assert raw["completion"]["status"] == expected
    record_run(tmp_path, cfg, result=EvaluationResult.from_payload("capture", cfg, raw))
    update = refresh_annotations(tmp_path)
    assert update["completion"]["status"] == expected
    assert update["coverage"]["samples_with_all_active_features_explained"] == 0
    assert read_run(tmp_path)["completion"]["status"] == expected
    assert list(read_samples(tmp_path))[0]["status"] == (
        "completed" if selected_available else "blocked"
    )


def test_constant_behavior_remains_undefined_after_annotation_refresh(
    mounted, monkeypatch, tmp_path
):
    import saebase.datasets.selection as selection

    from saebase.results import EvaluationResult
    from saebase.runtime.experiment_results import record_run
    from tasks.capture.annotations import read_samples, refresh_annotations

    cfg, pair = mounted
    cfg.task.prompt = ""
    cfg.task.sar.target_token_id = cfg.task.sar.reference_token_id = None
    rows = [
        {"sample_id": "constant", "prompt": "2 3 4", "target_token_id": 0, "reference_token_id": 1}
    ]
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(rows, {"selected_count": 1}),
    )
    monkeypatch.setattr(pair.model, "forward_logits", lambda **kw: torch.zeros(3))
    monkeypatch.setattr(pair.model, "patch_logits", lambda **kw: torch.zeros(3))
    raw = operation.execute(cfg)
    assert raw["counts"]["undefined"] == 1
    result = EvaluationResult.from_payload("capture", cfg, raw)
    assert (
        result.exit_code == 0
    )  # Evaluated mathematical undefined follows the existing result contract.
    record_run(tmp_path, cfg, result=result)
    update = refresh_annotations(tmp_path)
    assert update["completion"]["counts"]["undefined"] == 1
    assert list(read_samples(tmp_path))[0]["status"] == "undefined"


def test_replay_and_missing_patch_capability_are_rejected(mounted, monkeypatch):
    from saebase.api import UnsupportedCapability

    cfg, pair = mounted
    monkeypatch.setattr(pair.model, "patch_logits", lambda **kw: torch.ones(3) * 100)
    with pytest.raises(RuntimeError, match="Identity patch"):
        operation.execute(cfg)
    from dataclasses import replace

    monkeypatch.setattr(pair.model, "capabilities", replace(pair.model.capabilities, patch=False))
    with pytest.raises(UnsupportedCapability, match="patch"):
        operation.execute(cfg)


def test_undefined_sar_does_not_hide_incomplete_annotations(mounted, monkeypatch, tmp_path):
    import saebase.datasets.selection as selection

    from saebase.results import EvaluationResult
    from saebase.runtime.experiment_results import record_run
    from tasks.capture.annotations import refresh_annotations

    cfg, pair = mounted
    cfg.task.prompt = ""
    cfg.task.neuronpedia.enabled = True
    cfg.task.neuronpedia.allow_network = False
    rows = [{"sample_id": "constant", "prompt": "2 3 4"}]
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(rows, {"selected_count": 1}),
    )
    monkeypatch.setattr(pair.model, "forward_logits", lambda **kw: torch.zeros(3))
    monkeypatch.setattr(pair.model, "patch_logits", lambda **kw: torch.zeros(3))
    result = operation.execute(cfg)
    assert result["counts"]["blocked"] == 1 and result["completion"]["status"] == "incomplete"
    record_run(tmp_path, cfg, result=EvaluationResult.from_payload("capture", cfg, result))
    update = refresh_annotations(tmp_path)
    assert update["completion"]["status"] == "incomplete"


def test_registered_capture_can_configure_sar_without_import_or_computation(mounted, monkeypatch):
    from hydra import compose, initialize_config_dir

    from saebase.runtime.cfg.config_validate import validate_hydra_cfg
    from tasks.capture.task import CaptureTask

    config, _ = mounted
    Path(config.task.sar.source).write_text("raise RuntimeError('source must not run in dry-run')")
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=capture",
                "model=pythia70m",
                "task.layer=3",
                "task.mode=sar",
                f"task.sar.source={config.task.sar.source}",
            ],
        )
    monkeypatch.setattr(operation, "_pair", lambda cfg: pytest.fail("must not load weights"))
    validate_hydra_cfg(cfg)
    result = CaptureTask().run(cfg)
    assert result.status == "configured_only" and result.exit_code == 0


def test_benchmark_reuses_capture_estimator():
    from tasks.capture.sar import evaluate_sar
    from tasks.capture.sar import evaluate_sar as benchmark_sar

    assert benchmark_sar is evaluate_sar


def test_changed_external_estimator_prevents_completion_skip(mounted):
    from saebase.results import EvaluationResult
    from tasks.capture.task import CaptureTask

    cfg, _ = mounted
    result = EvaluationResult.from_payload("capture", cfg, operation.execute(cfg))
    task = CaptureTask()
    assert task.continuation_action(result) == "skip"
    source = Path(cfg.task.sar.source)
    source.write_text(source.read_text() + "\n# changed source identity\n")
    assert task.continuation_action(result) == "retry"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_affine_diagnostic_separates_model_rounding_from_decoder_nonlinearity(dtype):
    from dataclasses import replace

    from saebase.api import ActivationSite
    from saebase.api.features import FeatureSnapshot, edit_features
    from tasks.capture.sar import _check_affine_decode

    class DenseDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(17)
            self.weight = torch.nn.Parameter(torch.randn(8, 8, generator=generator))
            self.cfg = SimpleNamespace(d_in=8, d_sae=8)

        def encode(self, h):
            return h

        def decode(self, z):
            return z @ self.weight

    sae = DenseDecoder()
    generator = torch.Generator().manual_seed(32)
    codes = torch.randn(8, generator=generator)
    activation = torch.randn(8, generator=generator).to(dtype)
    snapshot = FeatureSnapshot(
        torch.tensor([[1]]),
        ActivationSite(0, "blocks.0.hook_resid_post"),
        activation,
        codes,
        sae.decode(codes).detach(),
    )
    before = activation.clone()
    _check_affine_decode(sae, snapshot, [0, 1, 2, 3])
    assert snapshot.activation.dtype == dtype and torch.equal(snapshot.activation, before)
    assert edit_features(sae, snapshot, replacements={0: 0.0}).activation.dtype == dtype
    original = sae.decode
    sae.decode = lambda z: original(z) + z.sum(-1, keepdim=True).square()
    with pytest.raises(ValueError, match="not additive"):
        _check_affine_decode(sae, snapshot, [0, 1, 2, 3])
    # The diagnostic promotes its own copy, leaving the actual capture untouched.
    assert replace(snapshot, activation=activation.float()).activation.dtype == torch.float32


def test_context_capability_failure_is_unsupported_in_batch(mounted, monkeypatch):
    import saebase.datasets.selection as selection

    from saebase.api import UnsupportedCapability

    cfg, _ = mounted
    cfg.task.prompt = ""
    rows = [{"sample_id": "long", "prompt": "2 3 4"}]
    monkeypatch.setattr(
        selection,
        "select_dataset",
        lambda *a, **kw: selection.SampleSelection(rows, {"selected_count": 1}),
    )

    def unsupported(*args, **kwargs):
        raise UnsupportedCapability("Complete prompt exceeds model context")

    monkeypatch.setattr(operation, "_capture", unsupported)
    raw = operation.execute(cfg)
    assert raw["counts"]["unsupported"] == 1 and raw["counts"]["failed"] == 0
