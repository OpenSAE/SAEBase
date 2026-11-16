"""Offline third-party extension API integration tests; no downloads or GPU."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from saebase.api import ActivationSite
from saebase.extensions import (
    ENTRY_POINT_GROUP,
    ExtensionBundle,
    ExternalMetric,
    ExternalModel,
    ExternalObservation,
    ExternalSAE,
    HookSpec,
    discover_extensions,
    external_pair_sources,
    list_extensions,
    load_external_pair,
    register_extension,
)
from saebase.extensions import registry as extensions_registry
from tasks.benchmark.execution.planner import build_plan
from tasks.benchmark.execution.runner import execute
from tasks.benchmark.execution.schema import BenchmarkConfig, MetricSpec
from tasks.benchmark.metrics import registry as benchmark_registry
from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE

MODEL = "ext.toy.model"
SAE = "ext.toy.sae"
METRIC = "ext.toy.score"
SITE = ActivationSite(0, "blocks.0.hook_resid_post")
SHA = "b" * 64


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """Keep third-party test registrations out of the global built-in registry."""
    for name in ("_MODELS", "_SAES", "_METRICS", "_METRIC_DIGESTS"):
        monkeypatch.setattr(extensions_registry, name, {})
    for name in ("_BUNDLES", "_DISCOVERED"):
        monkeypatch.setattr(extensions_registry, name, set())
    monkeypatch.setattr(benchmark_registry, "_METRICS", dict(benchmark_registry._METRICS))


def bundle(source_file: Path, *, use_codes: bool = False) -> ExtensionBundle:
    metric = ExternalMetric(
        spec=MetricSpec(
            metric_id=METRIC,
            version="1",
            required_inputs=("m1_codes",) if use_codes else ("external_pair",),
            sample_unit="dataset",
            direction="contextual",
            protocol_id="ext.toy.metric.v1",
        ),
        source="https://example.org/original/toy_metric.py",
        revision="demo-v1",
        source_file=source_file,
        evaluator=lambda ctx: ExternalObservation(0.25 if ctx.code_store is not None else 0.5, 1),
    )
    return ExtensionBundle(
        name="ext.toy.extension",
        models=(
            ExternalModel(
                MODEL,
                "https://example.org/toy_model",
                "model-v1",
                (HookSpec(SITE, 4),),
                lambda device: ResearchToyModel().eval(),
            ),
        ),
        saes=(
            ExternalSAE(
                SAE,
                MODEL,
                "https://example.org/toy_sae",
                "sae-v1",
                SITE,
                4,
                7,
                "toy_linear",
                "none",
                "float32",
                lambda device: ResearchToySAE().eval(),
            ),
        ),
        metrics=(metric,),
    )


def config(source: Path) -> BenchmarkConfig:
    return BenchmarkConfig.from_dict(
        {
            "schema": "saebase.benchmark.config.v1",
            "name": "third-party-offline",
            "targets": [
                {
                    "model_key": MODEL,
                    "sae_keys": [SAE],
                    "layer": 0,
                    "hook_name": SITE.hook_name,
                }
            ],
            "dataset": {
                "id": "toy-data",
                "samples": [{"sample_id": "toy1", "token_ids": [1, 2, 3], "split": "test"}],
            },
            "metrics": [METRIC],
            "sources": {**external_pair_sources(MODEL, SAE), METRIC: str(source)},
            "options": {"tokenizer_sha256": SHA},
        }
    )


def test_explicit_registry_and_model_sae_probe(tmp_path):
    source = tmp_path / "original_metric.py"
    source.write_text("def estimate(): return 0.5\n", encoding="utf-8")
    register_extension(bundle(source))
    assert list_extensions()["models"] == (MODEL,)
    assert list_extensions()["saes"] == (SAE,)
    assert list_extensions()["metrics"] == (METRIC,)
    unprobed = load_external_pair(model_key=MODEL, sae_key=SAE, layer=0)
    assert unprobed.verification == "loaded_unverified"
    assert unprobed.identity.hook_name == SITE.hook_name
    assert unprobed.identity.normalization == "none"
    assert unprobed.identity.d_in == 4
    assert unprobed.identity.d_sae == 7
    assert unprobed.identity.model_revision == "model-v1"
    assert unprobed.identity.revision == "sae-v1"
    tested = load_external_pair(
        model_key=MODEL,
        sae_key=SAE,
        layer=0,
        probe_tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )
    assert tested.verification == "probe_passed_not_checkpoint_certified"


def test_external_metric_executes_and_rechecks_original_source(tmp_path):
    source = tmp_path / "original_metric.py"
    source.write_text("def estimate(): return 0.5\n", encoding="utf-8")
    register_extension(bundle(source))
    cfg = config(source)
    assert build_plan(cfg)["metric_jobs"] == 1
    with pytest.raises(ValueError, match="real=True"):
        execute(cfg, tmp_path)
    rows = execute(cfg, tmp_path, real=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["metric_id"] == METRIC
    assert row["value"] == 0.5
    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row["source_commit"] == "demo-v1"
    assert row["metadata"]["verification"] == "metric_source_verified_not_model_certified"
    assert execute(cfg, tmp_path, real=True) == rows
    source.write_text("def estimate(): return 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        execute(cfg, tmp_path, real=True)


def test_external_metric_can_use_cached_codes(tmp_path):
    source = tmp_path / "code_metric.py"
    source.write_text("def estimate(): return 0.25\n", encoding="utf-8")
    register_extension(bundle(source, use_codes=True))
    cfg = config(source)
    plan = build_plan(cfg)
    assert any(node["operation"] == "m1_encode" for node in plan["nodes"])
    rows = execute(cfg, tmp_path, real=True)
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    assert rows[0]["value"] == 0.25


def test_source_path_mismatch_is_blocked_not_silently_completed(tmp_path):
    source = tmp_path / "code_metric.py"
    source.write_text("def estimate(): return 0.5\n", encoding="utf-8")
    register_extension(bundle(source))
    cfg_dict = copy.deepcopy(config(source).to_dict())
    cfg_dict["sources"][METRIC] = str(tmp_path / "different.py")
    cfg = BenchmarkConfig.from_dict(cfg_dict)
    result = execute(cfg, tmp_path, real=True)
    assert len(result) == 1
    assert result[0]["status"] == "blocked"
    assert "registered original source" in result[0]["reason"]


def test_hook_and_width_conflicts_rejected_before_factory(tmp_path):
    source = tmp_path / "original.py"
    source.write_text("x = 1\n", encoding="utf-8")
    original = bundle(source)
    with pytest.raises(ValueError, match="hook width"):
        register_extension(
            ExtensionBundle(
                "ext.toy.other",
                original.models,
                (
                    ExternalSAE(
                        SAE,
                        MODEL,
                        "https://example.org/sae",
                        "revision-1",
                        SITE,
                        5,
                        7,
                        "linear",
                        "none",
                        "float32",
                        lambda device: ResearchToySAE().eval(),
                    ),
                ),
            )
        )
    assert list_extensions()["models"] == ()
    with pytest.raises(ValueError, match="normalization"):
        ExternalSAE(
            SAE,
            MODEL,
            "https://example.org",
            "rev",
            SITE,
            4,
            7,
            "linear",
            "unspecified",
            "float32",
            lambda device: None,
        )


def test_registrations_cannot_override_builtin_or_repeat(tmp_path):
    source = tmp_path / "score.py"
    source.write_text("x = 1\n", encoding="utf-8")
    provided = bundle(source)
    register_extension(provided)
    with pytest.raises(ValueError, match="already registered"):
        register_extension(provided)
    assert list_extensions()["bundles"] == ("ext.toy.extension",)


def test_opt_in_discovery_and_no_implicit_execution(monkeypatch, tmp_path):
    source = tmp_path / "original.py"
    source.write_text("x=1\n", encoding="utf-8")
    calls = []

    class FakeEntry:
        name = "toy"

        def load(self):
            calls.append("load")
            return lambda: bundle(source)

    def fake_entry_points(*, group):
        assert group == ENTRY_POINT_GROUP
        return [FakeEntry()]

    monkeypatch.setattr(extensions_registry, "entry_points", fake_entry_points)
    assert calls == []
    with pytest.raises(ValueError, match="explicitly"):
        discover_extensions(())
    assert discover_extensions(["toy"]) == ("toy",)
    assert calls == ["load"]
    with pytest.raises(ValueError, match="already discovered"):
        discover_extensions(["toy"])


def test_external_provider_changes_cannot_reuse_frozen_run(tmp_path):
    source = tmp_path / "immutable_metric.py"
    source.write_text("x=1\n", encoding="utf-8")
    register_extension(bundle(source))
    cfg = config(source)
    original = external_pair_sources(MODEL, SAE)
    assert cfg.sources[SAE] == original[SAE]
    tampered = copy.deepcopy(cfg.to_dict())
    tampered["sources"][SAE]["revision"] = "new-checkpoint"
    new_cfg = BenchmarkConfig.from_dict(tampered)
    with pytest.raises(ValueError, match="external provenance changed"):
        build_plan(new_cfg)
    tampered = copy.deepcopy(cfg.to_dict())
    tampered["sources"][MODEL]["hooks"][0]["width"] = 128
    with pytest.raises(ValueError, match="external provenance changed"):
        build_plan(BenchmarkConfig.from_dict(tampered))


def test_qwen_external_checkpoint_when_requested():
    import os

    if os.getenv("SAEBASE_REAL_QWEN_PROBE") != "1":
        pytest.skip("Explicitly enable the pinned real Qwen checkpoint integration")
    import json

    import torch
    from transformers import AutoTokenizer

    from models.hf_hooks_backend import load_hf_hooks_model
    from saebase.api import ActivationSite, capture_features, edit_features
    from saebase.api.sae_loading import load_sae_for_profile
    from saebase.extensions import (
        ExtensionBundle,
        ExternalModel,
        ExternalSAE,
        HookSpec,
        load_external_pair,
        register_extension,
    )

    MODEL_REPO = "Qwen/Qwen3-1.7B-Base"
    MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    SAE_REPO = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
    SAE_REVISION = "ce1a79d9c5163932d65c417380e53230e1086370"
    LAYER = 7
    SITE = ActivationSite(LAYER, "blocks.7.hook_resid_post", position=-1)
    PROMPT = "The capital of France is"

    def load_model(device: str):
        handle = load_hf_hooks_model(
            MODEL_REPO,
            key="qwen3_1_7b_base",
            family="qwen",
            device=device,
            dtype=torch.float32,
            revision=MODEL_REVISION,
        )
        return handle.model

    def load_sae(device: str):
        sae, _meta = load_sae_for_profile(
            "qwen3_1_7b_base_resid_w32k",
            LAYER,
            device=device,
            revision=SAE_REVISION,
        )
        return sae

    def main() -> dict:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        register_extension(
            ExtensionBundle(
                name="ext.opensae.qwen3_1_7b_l7",
                models=(
                    ExternalModel(
                        key="ext.opensae.qwen3_1_7b_base",
                        source=f"https://huggingface.co/{MODEL_REPO}",
                        revision=MODEL_REVISION,
                        hooks=(HookSpec(SITE, 2048),),
                        factory=load_model,
                        backend="hf_hooks",
                    ),
                ),
                saes=(
                    ExternalSAE(
                        key="ext.opensae.qwen3_1_7b_base_l7",
                        model_key="ext.opensae.qwen3_1_7b_base",
                        source=f"https://huggingface.co/{SAE_REPO}",
                        revision=SAE_REVISION,
                        site=SITE,
                        d_in=2048,
                        d_sae=32768,
                        architecture="topk",
                        normalization="none",
                        dtype="float32",
                        sae_id="layer7.sae.pt",
                        factory=load_sae,
                    ),
                ),
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
        tokens = tokenizer(PROMPT, return_tensors="pt").input_ids
        if device == "cuda":
            tokens = tokens.cuda()

        pair = load_external_pair(
            model_key="ext.opensae.qwen3_1_7b_base",
            sae_key="ext.opensae.qwen3_1_7b_base_l7",
            layer=LAYER,
            device=device,
            probe_tokens=tokens,
        )
        snap = capture_features(pair.model, pair.sae, site=SITE, tokens=tokens)
        selected = int(torch.argmax(snap.codes.abs()).item())
        edited = edit_features(pair.sae, snap, replacements={selected: 0.0})
        delta = float(
            torch.linalg.vector_norm(edited.activation.float() - snap.activation.float()).item()
        )
        record = {
            "schema": "saebase.stage_d_recovery_03.external_interop.v1",
            "status": pair.verification,
            "claim": "adapter interoperability on this exact checkpoint and site only",
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "sae_repo": SAE_REPO,
            "sae_revision": SAE_REVISION,
            "layer": LAYER,
            "hook_name": SITE.hook_name,
            "d_in": 2048,
            "d_sae": 32768,
            "dtype": "float32",
            "normalization": "none",
            "l0": int(torch.count_nonzero(snap.codes).item()),
            "ablated_feature_id": selected,
            "activation_delta_l2": delta,
            "finite": bool(
                torch.isfinite(snap.reconstruction).all()
                and torch.isfinite(edited.activation).all()
            ),
            "nontrivial_ablation": delta > 1e-8,
        }
        if (
            pair.verification != "probe_passed_not_checkpoint_certified"
            or not record["nontrivial_ablation"]
        ):
            raise SystemExit(json.dumps(record, indent=2))
        return record

    result = main()
    assert result["status"] == "probe_passed_not_checkpoint_certified"
    assert result["model_revision"] == MODEL_REVISION
    assert result["sae_revision"] == SAE_REVISION
    assert result["finite"] and result["nontrivial_ablation"]
