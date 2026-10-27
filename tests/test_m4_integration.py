"""M4 checks M3/M1/M2 artifacts without performing extra model forwards."""

import json
from types import SimpleNamespace
import pytest

from saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ModelAdapter
from saebase.benchmarks import BenchmarkConfig
from saebase.benchmarks.runner import execute
from saebase.benchmarks.report import write_report
from saebase.benchmarks.planner import build_plan
from saebase.reliability import AuditConfig, evaluate_m3, write_audit, read_audit
from saebase.reliability.ranking import rank_stability
from saebase.reliability.schema import Observation


def base(two=False):
    return {
        "schema": "saebase.benchmark.config.v1",
        "name": "M4 grouped synthetic test",
        "targets": [
            {
                "model_key": "synthetic",
                "sae_keys": ["toy", "toy2"] if two else ["toy"],
                "layer": 0,
                "hook_name": "blocks.0.hook_resid_post",
            }
        ],
        "dataset": {
            "id": "synthetic-prompt-groups",
            "samples": [
                {
                    "sample_id": f"s{g}-{v}",
                    "prompt_group_id": f"group{g}",
                    "split": "test",
                    "token_ids": [1 + g, 2 + g, 3 + v],
                    "target_token_id": 0,
                    "reference_token_id": 1,
                }
                for g in range(4)
                for v in range(2)
            ],
        },
        "metrics": ["core.l0", "intervention.logit_delta_l2", "intervention.kl_baseline_to_edited"],
        "conditions": [
            {"name": "identity", "operation": "identity"},
            {"name": "ablate-0", "operation": "ablate", "feature_ids": [0]},
        ],
        "max_measurements": 100,
        "max_activation_rows": 100,
        "options": {"tokenizer_sha256": "b" * 64},
    }


def setup(tmp_path, two=False):
    cfg = BenchmarkConfig.from_dict(base(two))
    m = ModelAdapter(ResearchToyModel().eval())
    models = {"toy": ResearchToySAE().eval()}
    if two:
        second = ResearchToySAE().eval()
        with __import__("torch").no_grad():
            second.bias.add_(0.05)
        models["toy2"] = second
    calls = {"n": 0}
    original = m.model.run_with_cache

    def capture(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    m.model.run_with_cache = capture
    loader = lambda mk, sk, layer: SimpleNamespace(model=m, sae=models[sk])
    rows = execute(cfg, tmp_path, real=True, loader=loader)
    return cfg, rows, calls, m


def audit_cfg():
    return AuditConfig(
        evaluation_seeds=(0, 1, 2, 3, 4), bootstrap_replicates=60, sample_sizes=(2, 4, 32)
    )


def test_m4_audits_verified_m3_measures_and_embeds_report(tmp_path):
    cfg, rows, calls, _ = setup(tmp_path)
    root = tmp_path / cfg.run_id
    old = calls["n"]
    audit = evaluate_m3(root, audit_cfg())
    assert calls["n"] == old, "M4 must not trigger model forwards"
    assert audit["status"] == "completed"
    assert audit["model_forwards_during_audit"] == 0
    assert len(audit["metric_audits"]) == 5
    assert all(x["status"] == "completed" for x in audit["metric_audits"])
    assert all(x["audit"]["n_groups"] == 4 for x in audit["metric_audits"])
    assert all(x["audit"]["n_observations"] == 8 for x in audit["metric_audits"])
    assert write_audit(root, audit).exists()
    assert read_audit(root) == audit
    write_report(root, cfg, build_plan(cfg), rows)
    html = (root / "report.html").read_text()
    assert "M4 Reliability Audit" in html
    assert "no new model forwards" in html.lower()
    assert (root / "reliability_summary.csv").exists()
    again = evaluate_m3(root, audit_cfg())
    assert again == audit
    assert calls["n"] == old


def test_m4_runs_paired_comparison_on_matched_two_saes(tmp_path):
    cfg, rows, calls, _ = setup(tmp_path, two=True)
    audit = evaluate_m3(tmp_path / cfg.run_id, audit_cfg())
    assert len(audit["paired_comparisons"]) == 5
    assert all(x["status"] == "completed" for x in audit["paired_comparisons"])
    assert all(x["n_groups"] == 4 for x in audit["paired_comparisons"])
    assert calls["n"] > 0


def test_m4_rejects_corrupt_m2_source_shard(tmp_path):
    cfg, rows, _, _ = setup(tmp_path)
    candidates = list((tmp_path / "_cache" / "m2").rglob("shard-*.jsonl"))
    assert candidates
    candidates[0].write_text("{}\n")
    with pytest.raises(Exception, match="corrupt|sha|shard|identity|json"):
        evaluate_m3(tmp_path / cfg.run_id, audit_cfg())


def test_m4_audit_digest_and_m3_parent_digest(tmp_path):
    cfg, rows, _, _ = setup(tmp_path)
    root = tmp_path / cfg.run_id
    write_audit(root, evaluate_m3(root, audit_cfg()))
    (root / "reliability.json").write_text("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        read_audit(root)
    write_audit(root, evaluate_m3(root, audit_cfg()))
    manifest = json.loads((root / "metrics_manifest.json").read_text())
    manifest["sha256"] = "0" * 64
    (root / "metrics_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        read_audit(root)


def test_ranking_does_not_rank_contextual_metrics_or_unmatched_data():
    cfg = audit_cfg()

    def rows(sae, offset, metric):
        return [
            Observation("m", sae, metric, "d", "test", f"g{i}", f"s{i}", float(i + offset))
            for i in range(5)
        ]

    assert (
        rank_stability({"a": rows("a", 0, "core.l0"), "b": rows("b", 2, "core.l0")}, cfg)["status"]
        == "unsupported"
    )
    stable = rank_stability(
        {
            "a": rows("a", 0, "core.explained_variance"),
            "b": rows("b", 2, "core.explained_variance"),
        },
        cfg,
    )
    assert stable["status"] == "completed"
    assert stable["mean_kendall_tau"] == 1.0
    assert stable["mean_top_k_overlap"] == 1.0
    assert stable["ranking_reversal_frequency"] == 0.0


def test_recovery_of_core_mse_fvu_variance_without_model_forward(tmp_path):
    data = base(False)
    data["metrics"] = ["core.l0", "core.reconstruction_mse", "core.fvu", "core.explained_variance"]
    data["conditions"] = []
    cfg = BenchmarkConfig.from_dict(data)
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    counter = {"n": 0}
    original = model.model.run_with_cache

    def capture(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    model.model.run_with_cache = capture
    execute(
        cfg, tmp_path, real=True, loader=lambda mk, sk, layer: SimpleNamespace(model=model, sae=sae)
    )
    initial = counter["n"]
    report = evaluate_m3(
        tmp_path / cfg.run_id, audit_cfg(), sae_resolver=lambda model_key, sae_key, layer: sae
    )
    assert len(report["metric_audits"]) == 4
    assert all(x["status"] == "completed" for x in report["metric_audits"]), report
    assert all(x["audit"]["n_groups"] == 4 for x in report["metric_audits"])
    assert counter["n"] == initial, "SAE-only decode must not use the model"
    again = evaluate_m3(tmp_path / cfg.run_id, audit_cfg(), sae_resolver=lambda *args: sae)
    assert report == again
    with __import__("torch").no_grad():
        sae.bias.add_(1.0)
    with pytest.raises(ValueError, match="decoder checkpoint differs"):
        evaluate_m3(tmp_path / cfg.run_id, audit_cfg(), sae_resolver=lambda *args: sae)
