"""Work Plan, Atlas, human-intent and immutable-result regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.github.work_plan import lifecycle as life


def _write(root: Path, name: str, data: object) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def _fixture(root: Path) -> str:
    _write(root, "harness/core/project_graph.json", {"nodes": [{"id": "g.pipeline"}]})
    _write(root, "doc/knowledge/context_graph.json", {"nodes": [{"id": "q.sae.methods"}]})
    _write(root, "doc/knowledge/glossary.json", {"terms": [{"id": "term.sae.reproducible"}]})
    _write(root, "doc/knowledge/atlas_terms.json", {"terms": [{"id": "term.rclone"}]})
    _write(root, "saebase/runner.py", "def run_task():\n    return None\n")
    _write(root, "conf/experiment/example.yaml", "experiment:\n  name: example\n")
    human = {
        "schema": "saebase.from_human.v1",
        "plan_id": "WP23",
        "goal": "An observed and reproducible benchmark",
        "scientific_motivation": "Evidence for the research question",
    }
    _write(
        root,
        "harness/github/work_plan/from_human/WP23.md",
        "```from-human-json\n" + json.dumps(human) + "\n```\n",
    )
    manifest = {
        "schema": "saebase.workplan.v1",
        "plan_id": "WP23",
        "status": "reviewed",
        "from_human": "harness/github/work_plan/from_human/WP23.md",
        "atlas": {
            "core_ids": ["g.pipeline"],
            "knowledge_ids": ["q.sae.methods"],
            "context_term_ids": ["term.rclone"],
            "code_refs": ["saebase/runner.py"],
        },
        "experiment": {"study_id": "study23", "config": "conf/experiment/example.yaml"},
        "steps": [
            {
                "id": "setup",
                "needs": [],
                "permission": "local",
                "goal": "Observe setup",
                "evidence": "local receipt",
            },
            {
                "id": "run",
                "needs": ["setup"],
                "permission": "gpu",
                "goal": "Run frozen benchmark",
                "evidence": "model receipt",
            },
        ],
        "deliverables": [
            {"experiment": "E1", "metric": "accuracy", "expected": "100 real samples"}
        ],
    }
    path = "harness/github/work_plan/23_example.md"
    _write(root, path, "# WP23\n\n```workplan-json\n" + json.dumps(manifest) + "\n```\n")
    return path


def test_plan_from_human_atlas_and_filled_result(tmp_path):
    plan = _fixture(tmp_path)
    assert life.check(plan, root=tmp_path, check_contracts=False)["plan_id"] == "WP23"
    incomplete = life.completed(
        plan,
        {"schema": "saebase.workplan.run.v1", "plan_id": "WP23", "run_id": "r1", "results": {}},
        root=tmp_path,
    )
    assert "| E1 | accuracy | 100 real samples |  | NOT_RUN |" in incomplete
    receipt = {
        "schema": "saebase.workplan.run.v1",
        "plan_id": "WP23",
        "run_id": "r2",
        "source_commit": "a" * 40,
        "results": {
            "E1": {
                "status": "PASS",
                "value": 0.8,
                "notes": "100 actual samples",
                "evidence": "metrics/acc.json",
            }
        },
    }
    assert "0.8 | PASS" in life.completed(plan, receipt, root=tmp_path)
    receipt["results"]["E1"]["evidence"] = ""
    with pytest.raises(life.PlanError, match="truthful"):
        life.completed(plan, receipt, root=tmp_path)


def test_invalid_atlas_code_and_dag_fail_closed(tmp_path):
    plan = _fixture(tmp_path)
    path = tmp_path / plan
    original = path.read_text()
    for old, new in (
        ('"g.pipeline"', '"g.missing"'),
        ('"saebase/runner.py"', '"../escape.py"'),
        ('"needs": ["setup"]', '"needs": ["absent"]'),
    ):
        path.write_text(original.replace(old, new))
        with pytest.raises(life.PlanError):
            life.check(plan, root=tmp_path, check_contracts=False)
    path.write_text(original)
    (tmp_path / "harness/github/work_plan/from_human/WP23.md").unlink()
    with pytest.raises(life.PlanError):
        life.check(plan, root=tmp_path, check_contracts=False)
