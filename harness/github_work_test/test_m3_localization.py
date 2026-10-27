"""P3 Localize benchmark/curves/M4 acceptance on synthetic controlled input grids."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ModelAdapter
from saebase.data import BinaryFactor, ControlledInputDesign
from saebase.benchmarks import BenchmarkConfig, build_plan
from saebase.benchmarks.adapters.localization import (
    read_localization_artifact,
    grid_from_artifact,
    validate_localization_result,
)
from saebase.benchmarks.runner import execute, validate_external_sources
from saebase.benchmarks.report import write_report
from saebase.reliability import AuditConfig, evaluate_m3, write_audit, read_audit


class NativeContract:
    """Checks that SAEBase delegates, without duplicating Localize's algorithms."""

    __version__ = "contract-v1"

    def __init__(self):
        self.exact_calls = 0
        self.paired_seeds = []

    def exact_scope_curve(self, z, n_factors):
        self.exact_calls += 1
        assert z.shape == (8, 7) and n_factors == 3
        variance = float(np.var(z, axis=0).sum())
        assert variance > 0
        curve = np.zeros(z.shape[1] * n_factors + 1)
        curve[0] = variance
        curve[1] = 0.5 * variance

        class ScopeResult:
            total_variance = variance

            def budget(self, tolerance):
                assert tolerance == pytest.approx(0.1 * variance)
                return 2

            def assignment(self, budget):
                assert budget == 2
                return np.asarray([1, 2, 0, 0, 0, 0, 0], dtype=int)

        result = ScopeResult()
        result.curve = curve
        return result

    def estimated_scope_curve(self, z, pairs_per_scope, seed):
        assert pairs_per_scope == 32
        self.paired_seeds.append(seed)
        W = float(np.var(z, axis=0).sum())
        estimated = np.zeros(z.shape[1] * 3 + 1)
        estimated[0] = W
        estimated[1] = W * (0.2 + seed / 100)
        return estimated


def _design():
    return ControlledInputDesign.from_renderer(
        name="P3 toy design",
        factors=(
            BinaryFactor("first", ("1", "2")),
            BinaryFactor("second", ("3", "4")),
            BinaryFactor("third", ("5", "6")),
        ),
        render=lambda x: f"{x['first']} {x['second']} {x['third']}",
    )


def _config(two=False, metrics=None, pairs=32):
    model = ModelAdapter(ResearchToyModel().eval())
    design = _design()
    records = design.sample_records(model)
    payload = {
        "schema": "saebase.benchmark.config.v1",
        "name": "Localization P3 acceptance",
        "targets": [
            {
                "model_key": "synthetic",
                "layer": 0,
                "hook_name": "blocks.0.hook_resid_post",
                "sae_keys": ["sae-a", "sae-b"] if two else ["sae-a"],
            }
        ],
        "dataset": {"id": "P3 synthetic factorial", "samples": [r.to_dict() for r in records]},
        "metrics": metrics
        or ["localization.budget_at_90pct", "localization.budget_fraction_at_90pct"],
        "conditions": [],
        "max_activation_rows": 32,
        "options": {
            "tokenizer_sha256": hashlib.sha256(b"synthetic-toy-tokenizer").hexdigest(),
            "localization": {
                "name": design.name,
                "factors": [f.to_dict() for f in design.factors],
                "sample_ids_by_state": [s.sample_id for s in design.states],
                "pairs_per_scope": pairs,
            },
        },
    }
    return BenchmarkConfig.from_dict(payload)


def _loader(two=False):
    model = ModelAdapter(ResearchToyModel().eval())
    counts = {"forward_calls": 0}
    original = model.model.run_with_cache

    def counted(*args, **kw):
        counts["forward_calls"] += 1
        return original(*args, **kw)

    model.model.run_with_cache = counted
    saes = {"sae-a": ResearchToySAE().eval()}
    if two:
        other = ResearchToySAE().eval()
        with torch.no_grad():
            other.bias.add_(0.1)
        saes["sae-b"] = other
    return lambda mk, sk, layer: SimpleNamespace(model=model, sae=saes[sk]), counts


def test_localization_preflight_and_contracts():
    cfg = _config(two=True)
    plan = build_plan(cfg)
    assert plan["planned_activation_rows"] == 8
    assert plan["metric_jobs"] == 4
    assert sum(node["operation"] == "original_localize_eval" for node in plan["nodes"]) == 2
    assert BenchmarkConfig.from_dict(cfg.to_dict()).run_id == cfg.run_id
    base = cfg.to_dict()
    cases = [
        ("sample_ids_by_state", [base["dataset"]["samples"][0]["sample_id"]] * 8, "exactly once"),
        (
            "sample_ids_by_state",
            [x["sample_id"] for x in base["dataset"]["samples"][:7]],
            "exactly once",
        ),
        ("pairs_per_scope", 0, "positive"),
    ]
    for key, value, match in cases:
        d = copy.deepcopy(base)
        d["options"]["localization"][key] = value
        with pytest.raises(ValueError, match=match):
            build_plan(BenchmarkConfig.from_dict(d))
    d = copy.deepcopy(base)
    d["dataset"]["samples"][0]["selection"] = {"mode": "all_valid"}
    with pytest.raises(ValueError, match="exactly one patch position"):
        build_plan(BenchmarkConfig.from_dict(d))
    d = copy.deepcopy(base)
    d["dataset"]["samples"][0]["prompt_group_id"] = "fake-independent-group"
    with pytest.raises(ValueError, match="one prompt group"):
        build_plan(BenchmarkConfig.from_dict(d))
    d = copy.deepcopy(base)
    d["dataset"]["samples"][0]["token_ids"] = d["dataset"]["samples"][1]["token_ids"]
    with pytest.raises(ValueError, match="identical token IDs"):
        build_plan(BenchmarkConfig.from_dict(d))


def test_localization_full_pipeline_and_reliability(tmp_path: Path):
    cfg = _config(two=True)
    stub = NativeContract()
    loader, counts = _loader(two=True)
    rows = execute(cfg, tmp_path, real=True, loader=loader, localization_paper=stub)
    assert counts["forward_calls"] == 8, "second SAE must reuse identical M1 activations"
    assert stub.exact_calls == 2
    assert len(rows) == 4 and all(row["status"] == "completed" for row in rows)
    budgets = [r for r in rows if r["metric_id"] == "localization.budget_at_90pct"]
    fractions = [r for r in rows if r["metric_id"] == "localization.budget_fraction_at_90pct"]
    assert [r["value"] for r in budgets] == [2.0, 2.0]
    assert [r["value"] for r in fractions] == pytest.approx([2 / 21, 2 / 21])
    assert all(r["n_valid"] == r["n_requested"] == 1 for r in rows)
    root = tmp_path / cfg.run_id
    assert "Input Localization Curves" in (root / "report.html").read_text()
    assert "localization/" in (root / "report.html").read_text()
    artifacts = {}
    for row in budgets:
        artifact = validate_localization_result(root, cfg, row, paper=stub)
        assert artifact["curve"] and artifact["assignment"]
        assert artifact["design_manifest"]["schema"] == "saebase.controlled_inputs.v1"
        assert len(artifact["curve"]) == 22
        assert len(artifact["paired_estimated_curve"]) == 22
        assert artifact["source_sha256"] == row["source_sha256"]
        assert len(row["source_sha256"]) == 64
        assert grid_from_artifact(root, artifact).codes.shape == (8, 7)
        artifacts[row["sae_id"]] = artifact

    audit_config = AuditConfig(
        evaluation_seeds=(0, 1, 2, 3),
        bootstrap_replicates=30,
        metric_ids=("localization.budget_at_90pct",),
    )
    initial = counts["forward_calls"]
    report = evaluate_m3(root, audit_config, localization_paper=stub)
    assert counts["forward_calls"] == initial, "M4 must not run model forwards"
    assert not report["metric_audits"] and not report["paired_comparisons"]
    assert len(report["localization_audits"]) == 2
    assert len(report["localization_comparisons"]) == 1
    assert report["localization_comparisons"][0]["status"] == "completed"
    assert report["localization_comparisons"][0]["population_confidence_interval"] is None
    assert report["localization_comparisons"][0]["ranking_claim"] is None
    for audit in report["localization_audits"]:
        assert audit["audit"]["n_groups"] == 1
        assert audit["audit"]["sampling_stability"]["status"] == "completed"
        assert audit["audit"]["sampling_stability"]["n_evaluation_seeds"] == 4
        assert audit["audit"]["sampling_stability"]["population_confidence_interval"] is None
        assert len(audit["audit"]["sampling_stability"]["seed_results"]) == 4
    assert write_audit(root, report).is_file()
    assert read_audit(root) == report
    write_report(root, cfg, build_plan(cfg), rows)
    html = (root / "report.html").read_text()
    assert "Matched Localize SAE comparisons" in html
    assert "not a confidence interval" in html
    assert (root / "reliability_summary.csv").exists()
    again = execute(cfg, tmp_path, real=True, loader=loader, localization_paper=stub, resume=True)
    assert again == rows and counts["forward_calls"] == initial


def test_corrupt_curve_or_wrong_source_is_rejected(tmp_path):
    cfg = _config()
    stub = NativeContract()
    loader, _ = _loader()
    rows = execute(cfg, tmp_path, real=True, loader=loader, localization_paper=stub)
    root = tmp_path / cfg.run_id
    ref = rows[0]["measurement_refs"][0]
    path = root / ref
    before = path.read_text()
    item = json.loads(before)
    item["curve"][0] += 0.5
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="integrity"):
        validate_external_sources(cfg, rows, root=root, localization_paper=stub)
    path.write_text(before)
    assert validate_external_sources(cfg, rows, root=root, localization_paper=stub) is None

    # A function loaded from a genuinely different source file has a new digest.
    import importlib.util

    source = tmp_path / "revised_localize.py"
    source.write_text(
        "def exact_scope_curve(z, n_factors):" + chr(10) + "    return None" + chr(10)
    )
    spec = importlib.util.spec_from_file_location("revised_localize", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="estimator changed"):
        validate_external_sources(cfg, rows, root=root, localization_paper=module)
    with pytest.raises(ValueError, match="artifact reference"):
        read_localization_artifact(root, "../../arbitrary.json")


def test_cross_sae_mismatch_is_not_compared():
    from saebase.reliability.localization import compare_localization

    cfg = _config(two=True)
    data = {
        "design_manifest": {"design_key": "abc", "factors": [1], "states": [2]},
        "source_sha256": "a",
        "n_factors": 3,
        "n_features": 7,
        "relative_omission": 0.1,
        "budget_at_90pct": 2,
        "budget_fraction_at_90pct": 2 / 21,
    }
    audit = {"audit": {"sampling_stability": {"status": "not_requested"}}}
    first = {
        "model_id": "synthetic",
        "sae_id": "sae-a",
        "metric_id": "localization.budget_at_90pct",
        "dataset_id": "d",
        "split": "test",
        "site": {"layer": 0, "hook_name": "blocks.0.hook_resid_post", "position": -1},
    }
    second = {**first, "sae_id": "sae-b"}
    diff = copy.deepcopy(data)
    diff["design_manifest"]["design_key"] = "changed"
    result = compare_localization([(first, data, audit), (second, diff, audit)])
    assert len(result) == 1 and result[0]["status"] == "unsupported"


def test_original_localize_native_when_installed(tmp_path):
    """Optional exact source acceptance with untouched SAE-Input-Localization."""
    original = pytest.importorskip("sae_localize")
    cfg = _config(metrics=["localization.budget_at_90pct"], pairs=None)
    loader, _ = _loader()
    rows = execute(cfg, tmp_path, real=True, loader=loader)
    assert rows[0]["status"] == "completed", rows
    art = read_localization_artifact(tmp_path / cfg.run_id, rows[0]["measurement_refs"][0])
    grid = grid_from_artifact(tmp_path / cfg.run_id, art)
    native = original.exact_scope_curve(grid.codes, n_factors=3)
    np.testing.assert_allclose(art["curve"], native.curve, atol=1e-10, rtol=0)
    assert art["budget_at_90pct"] == native.budget(0.1 * native.total_variance)
