"""Offline evidence validation: no synthetic fixtures are counted as real weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.benchmark.coverage_contract import load_coverage, verify_cpu_sae_evidence
from saebase.tools.phase_bc_aggregate import aggregate

CATALOG = Path(__file__).resolve().parents[3] / "conf" / "coverage" / "stage_bc.json"


def _synthetic_evidence():
    catalog = load_coverage(CATALOG)
    for entry in catalog.entries:
        for layer in entry.cpu_probe_layers:
            yield {
                "schema": "saebase.coverage.cpu_sae_acceptance.v1",
                "status": "sae_checkpoint_verified_only",
                "model_sae_pair_verified": False,
                "model_loaded": False,
                "sae_key": entry.sae_key,
                "model_key": entry.model_key,
                "layer": layer,
                "hook_name": entry.source(layer)["hook_name"],
                "source": entry.source(layer),
                "sae_state_sha256": "a" * 64,
                "d_in": 1024,
                "d_sae": 16384,
                "synthetic_probe_l0": [32, 30],
                "synthetic_probe_reconstruction_mse": 1.2,
            }


def _write(folder: Path, data: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{data['sae_key']}_l{data['layer']}.json").write_text(json.dumps(data))


def test_accept_only_complete_nonfake_provenance_shape():
    catalog = load_coverage(CATALOG)
    entry = next(_synthetic_evidence())
    summary = verify_cpu_sae_evidence(catalog, entry)
    assert summary["status"] == "sae_checkpoint_verified_only"
    assert summary["sae_state_sha256"] == "a" * 64
    for key, value, match in [
        ("sae_state_sha256", "short", "SHA256"),
        ("hook_name", "wrong", "source identity"),
        ("model_sae_pair_verified", True, "SAE-only"),
        ("synthetic_probe_reconstruction_mse", float("nan"), "finite"),
        ("d_sae", 0, "dimensions"),
    ]:
        bad = dict(entry)
        bad[key] = value
        with pytest.raises(ValueError, match=match):
            verify_cpu_sae_evidence(catalog, bad)


def test_aggregate_21_real_artifact_contracts_and_idempotent_repair(tmp_path):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    all_entries = list(_synthetic_evidence())
    assert len(all_entries) == 21
    failing = all_entries[6]
    for d in all_entries:
        if d is failing:
            d = {
                "schema": "saebase.coverage.cpu_sae_acceptance.v1",
                "status": "blocked",
                "sae_key": d["sae_key"],
                "layer": d["layer"],
                "error_type": "ValueError",
                "model_sae_pair_verified": False,
                "model_loaded": False,
            }
        _write(original, d)
    partial = aggregate(original, CATALOG)
    assert partial["expected"] == 21
    assert partial["sae_only_verified"] == 20
    assert partial["failed_or_blocked"] == 1
    assert not partial["all_checkpoints_verified"]
    _write(replacement, failing)
    corrected = aggregate(original, CATALOG, replacement)
    assert corrected["sae_only_verified"] == 21
    assert corrected["all_checkpoints_verified"]
    assert len(corrected["entries"]) == 21
    repaired = next(
        x
        for x in corrected["entries"]
        if x["sae_key"] == failing["sae_key"] and x["layer"] == failing["layer"]
    )
    assert repaired["repaired_after_block"]


def test_aggregate_rejects_silent_weight_replacement(tmp_path):
    original = tmp_path / "baseline"
    repaired = tmp_path / "replacements"
    item = next(_synthetic_evidence())
    _write(original, item)
    conflicting = {**item, "sae_state_sha256": "b" * 64}
    _write(repaired, conflicting)
    with pytest.raises(ValueError, match="previously verified"):
        aggregate(original, CATALOG, repaired)


def test_aggregate_rejects_missing_or_duplicate_evidence(tmp_path):
    baseline = tmp_path / "baseline"
    item = next(_synthetic_evidence())
    _write(baseline, item)
    incomplete = aggregate(baseline, CATALOG)
    assert incomplete["missing"] == 20
    duplicate_folder = baseline / "duplicate"
    _write(duplicate_folder, item)
    with pytest.raises(ValueError, match="duplicated"):
        aggregate(baseline, CATALOG)


def test_gemma_9b_uses_an_upstream_mapped_real_checkpoint():
    from saebase.api.sae_profiles import get_profile

    key = "gemma2_9b_pt_scope_resid"
    assert get_profile(key).sae_id(20) == "layer_20/width_16k/average_l0_58"
    catalog = load_coverage(CATALOG)
    assert catalog.find(key).source(20)["source_file_id"].endswith("average_l0_58")


def test_published_real_stage_b_and_c_ledgers_are_consistent():
    import json
    import re

    root = Path(__file__).resolve().parents[3] / "conf" / "coverage"
    catalog = load_coverage(CATALOG)
    stage_c = json.loads((root / "stage_c_cpu_verified.json").read_text())
    expected = {(e.sae_key, layer) for e in catalog.entries for layer in e.cpu_probe_layers}
    records = stage_c["entries"]
    assert stage_c["checkpoint_count"] == 21 == len(records)
    assert {(r["sae_key"], r["layer"]) for r in records} == expected
    assert stage_c["distinct_sae_profiles"] == 17
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", r["sae_state_sha256"])
        and re.fullmatch(r"[0-9a-f]{64}", r["evidence_json_sha256"])
        and str(r["source_run"]) in stage_c["source_commits"]
        and str(r["source_run"]) in r["artifact_url"]
        for r in records
    )
    assert stage_c["repaired"]["old_id"].endswith("average_l0_57")
    assert stage_c["repaired"]["corrected_id"].endswith("average_l0_58")
    stage_b = json.loads((root / "stage_b_cpu_verified.json").read_text())
    pairs = stage_b["results"]
    assert len(pairs) == 3
    assert {r["sae_key"] for r in pairs} == {
        "pythia410m_mlp_topk",
        "qwen3_1_7b_base_resid_w32k",
        "r1_distill_qwen15b_resid_l20",
    }
    assert all(
        r["pair_verified"]
        and r["m3m4_verified"]
        and r["benchmark_metric_rows"] == 5
        and r["reliability_audits"] == 3
        and r["sample_count"] == 2
        and re.fullmatch(r"[0-9a-f]{64}", r["m3_sha256"])
        and re.fullmatch(r"[0-9a-f]{64}", r["m4_sha256"])
        for r in pairs
    )
