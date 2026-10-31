"""WP22 independent frozen-protocol and negative-gate acceptance checks.

No HF network access, paid APIs, training, GPU model downloads or publication.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments.qa500.runner import make_plan
from scripts.experiments.qa500.capture import site_list
from scripts.experiments.qa500.cohort import DATASETS, N_PER_DATASET, PROTOCOL_ID
from tasks.benchmark.qa_scoring import score_choices
from saebase.integrations.neuronpedia import annotate
from scripts.experiments.qa500.report import audit

ROOT = Path(__file__).resolve().parents[2]
PLAN = "harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md"


def test_full_matrix_matches_immutable_protocol_config():
    from scripts.experiments.qa500.context import get_protocol, get_dataset_ids, get_sites, get_tasks

    plan = make_plan()
    cfg = get_protocol()
    datasets = get_dataset_ids()
    sites = get_sites()
    tasks = get_tasks()
    assert cfg["protocol_id"] == PROTOCOL_ID
    assert cfg["seed"] == 20261031
    assert list(datasets) == list(DATASETS)
    assert cfg["samples_per_dataset"] == N_PER_DATASET == 500
    assert len(datasets) == 10
    assert len(sites) == 10
    assert len(site_list()) == 20
    assert tasks["execution"]["expected_model_qa_baselines"] == plan["expected_model_qa_baselines"]
    assert tasks["execution"]["expected_capture_rows"] == plan["expected_capture_rows"]
    assert (
        tasks["execution"]["expected_benchmark_summaries"]
        == plan["expected_benchmark_summaries"]
        == 200
    )
    assert tasks["execution"]["expected_intervene_rows"] == plan["expected_intervention_rows"]
    assert tasks["execution"]["expected_generate_rows"] == plan["expected_generation_rows"]
    assert tasks["execution"]["hf_publish_automatic"] is False
    from scripts.experiments.qa500.capture import SITES

    assert [(x["model_key"], x["sae_key"], x["layers"]) for x in sites] == [
        (model, sae, list(layers)) for model, sae, layers in SITES
    ]


def test_workplan_uses_only_preexisting_universal_harness_contracts():
    from harness.contracts.registry import declaration, resolve, catalog

    selected = declaration(PLAN)
    cases = selected["cases"]
    assert len(cases) == 7
    assert set(cases) == {
        "Q01_PREPARE",
        "Q02_PILOT",
        "Q03_CORE",
        "Q04_EFFECTS",
        "Q05_NATIVE",
        "Q06_AUDIT",
        "Q07_PUBLISH",
    }
    known = {r["id"] for r in catalog()["contracts"]}
    for case in cases:
        chosen = resolve(PLAN, case)
        assert set(chosen["contract_ids"]).issubset(known)
        assert set(catalog()["required_always"]).issubset(chosen["contract_ids"])
        assert "CT-TRACE" in chosen["contract_ids"]


def test_unmapped_external_explanations_do_not_become_scientific_facts():
    explanation = annotate(
        11,
        site_id="unmapped-site",
        sae_sha256="0" * 64,
        mapping={},
        annotations={},
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
    from tests.workplan.wp22.test_execution import MinimalPair

    p = MinimalPair()
    with pytest.raises(ValueError, match="unique"):
        score_choices(p, p.model.tokens(prompt="Question: testing"), ["A", "A"])


@pytest.mark.parametrize(
    ("dataset_id", "example", "config"),
    [
        (
            "commonsense_qa",
            {
                "question": "Which action?",
                "choices": {"label": ["A", "B"], "text": ["Walk", "Fly"]},
                "answerKey": "A",
            },
            None,
        ),
        ("bbh", {"input": "Is it true?", "target": "yes"}, "boolean_expressions"),
        ("math500", {"problem": "Compute 1+2", "answer": "3"}, None),
        (
            "medqa",
            {
                "question": "Which therapy?",
                "options": {"A": "first", "B": "second"},
                "answer_idx": "B",
            },
            None,
        ),
        (
            "hotpot_qa",
            {
                "_id": "hotpot-1",
                "question": "Who?",
                "answer": "Ada",
                "context": {
                    "title": ["Source One"],
                    "sentences": [["Ada", "was", "a", "pioneer."]],
                },
            },
            "distractor",
        ),
    ],
)
def test_all_remaining_real_source_shapes_are_normalized(dataset_id, example, config):
    from scripts.experiments.qa500.cohort import normalize, render_prompt

    record = normalize(DATASETS[dataset_id], example, source_config=config, source_index=0)
    prompt = render_prompt(record)
    assert record["question"]
    assert record["answer"]
    assert "Question:" in prompt
    assert "Answer: " + record["answer"] not in prompt
    assert record["prompt_sha256"]


def test_bootstrap_core_benchmark_is_reproducible_and_constant_cases():
    from scripts.experiments.qa500.measure import summarize_metric

    metric = summarize_metric([4.0] * 500)
    assert metric["value"] == 4.0
    assert metric["ci_lower_95"] == 4.0
    assert metric["ci_upper_95"] == 4.0
    assert metric["n_valid"] == 500
    assert summarize_metric([4.0] * 500) == metric
    with pytest.raises(ValueError):
        summarize_metric([float("nan")])


def test_architecture_has_no_experiment_entrypoints_under_datasets():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert not list((root / "datasets").glob("qa_*.py"))
    assert not (root / "harness/experiments/qa500_execute.sh").exists()
    assert (root / "scripts/experiments/qa500.sh").is_file()
    assert (root / "conf/experiment/qa500/protocol.json").is_file()
    assert (root / "scripts/experiments/qa500/runner.py").is_file()
    assert not (root / "experiments/qa500").exists()

    for folder in ("datasets", "models", "tasks", "saebase"):
        for file in (root / folder).rglob("*.py"):
            syntax = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            for obj in ast.walk(syntax):
                if isinstance(obj, ast.Import):
                    assert all(
                        not alias.name.startswith("scripts.experiments.qa500") for alias in obj.names
                    ), str(file)
                elif isinstance(obj, ast.ImportFrom):
                    assert not (obj.module or "").startswith("scripts.experiments.qa500"), str(file)


def test_run_paths_use_existing_task_artifact_layout(tmp_path):
    from scripts.experiments.qa500.context import task_dir, typed_file, study_dir

    path = task_dir(
        tmp_path,
        task="capture",
        dataset="mmlu_pro",
        model="gemma2_2b_base",
        sae_key="gemma2_2b_pt_l25_canonical",
        layer=13,
        initialize=True,
    )
    assert path.is_relative_to(tmp_path / "experiment" / "mmlu_pro")
    assert "/capture/qa500/" in path.as_posix()
    assert path.name == "qa500_20261031"
    assert typed_file(path, "metrics", "capture.jsonl").parent == path / "metrics"
    assert (path / "metadata" / "run.json").is_file()
    assert study_dir(tmp_path).is_relative_to(tmp_path / "experiment" / "_studies")


def test_independent_qa_sources_have_native_hydra_dataset_configs():
    from omegaconf import OmegaConf
    from scripts.experiments.qa500.context import get_dataset_ids
    from saebase.registry import get_qa_dataset_adapter

    root = Path(__file__).resolve().parents[2]
    for dataset_id in get_dataset_ids():
        adapter = get_qa_dataset_adapter(dataset_id)
        config = root / "conf" / "dataset" / (dataset_id + ".yaml")
        assert config.is_file(), config
        parsed = OmegaConf.load(config)
        assert parsed.dataset.name
        assert adapter.SPEC.key == dataset_id
