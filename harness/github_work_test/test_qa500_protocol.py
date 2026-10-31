"""WP22 independent frozen-protocol and negative-gate acceptance checks.

No HF network access, paid APIs, training, GPU model downloads or publication.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from saebase.datasets.qa_batch import make_plan
from saebase.datasets.qa_pipeline import site_list
from saebase.datasets.qa_standard import DATASETS, N_PER_DATASET, PROTOCOL_ID
from saebase.datasets.qa_inference import score_choices
from saebase.datasets.qa_neuronpedia import annotate
from saebase.datasets.qa_report import audit

ROOT = Path(__file__).resolve().parents[2]
PLAN = "harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md"


def test_full_matrix_matches_immutable_protocol_config():
    plan = make_plan()
    cfg = json.loads((ROOT / "conf/qa500_standard_v1.json").read_text())
    assert cfg["protocol_id"] == PROTOCOL_ID
    assert cfg["seed"] == 20261031
    assert cfg["datasets"] == list(DATASETS)
    assert cfg["dataset_count"] == len(DATASETS) == 10
    assert cfg["samples_per_dataset"] == N_PER_DATASET == 500
    assert cfg["model_count"] == 10
    assert cfg["site_count"] == len(site_list()) == 20
    assert cfg["execution"]["expected_model_qa_baselines"] == plan[
        "expected_model_qa_baselines"
    ]
    assert cfg["execution"]["expected_capture_rows"] == plan["expected_capture_rows"]
    assert cfg["execution"]["expected_intervene_rows"] == plan[
        "expected_intervention_rows"
    ]
    assert cfg["execution"]["expected_generate_rows"] == plan[
        "expected_generation_rows"
    ]
    assert cfg["execution"]["hf_publish_automatic"] is False
    expected = [(m["model_key"], m["sae_key"], m["layers"]) for m in cfg["models"]]
    from saebase.datasets.qa_pipeline import SITES

    actual = [(name, sae, list(layers)) for name, sae, layers in SITES]
    assert expected == actual


def test_workplan_uses_only_preexisting_universal_harness_contracts():
    from harness.contracts.registry import declaration, resolve, catalog

    selected = declaration(PLAN)
    cases = selected["cases"]
    assert len(cases) == 7
    assert set(cases) == {
        "Q01_PREPARE", "Q02_PILOT", "Q03_CORE", "Q04_EFFECTS",
        "Q05_NATIVE", "Q06_AUDIT", "Q07_PUBLISH",
    }
    known = {r["id"] for r in catalog()["contracts"]}
    for case in cases:
        chosen = resolve(PLAN, case)
        assert set(chosen["contract_ids"]).issubset(known)
        assert set(catalog()["required_always"]).issubset(chosen["contract_ids"])
        assert "CT-TRACE" in chosen["contract_ids"]


def test_unmapped_external_explanations_do_not_become_scientific_facts():
    explanation = annotate(
        11, site_id="unmapped-site", sae_sha256="0" * 64,
        mapping={}, annotations={},
    )
    assert explanation["explanation"] is None
    assert explanation["explanation_status"] == "unmapped"


def test_audit_fails_closed_on_empty_records(tmp_path):
    result = audit(tmp_path / "inputs", tmp_path / "results")
    assert result["status"] == "partial"
    assert result["counts"]["input_datasets_complete"] == 0
    assert result["counts"]["capture_complete"] == 0
    assert result["defect_count"] >= 10


def test_qa_answer_scorer_rejects_repeated_choices():
    from harness.github_work_test.test_qa500_execution import MinimalPair

    p = MinimalPair()
    with pytest.raises(ValueError, match="unique"):
        score_choices(p, p.model.tokens(prompt="Question: testing"), ["A", "A"])
