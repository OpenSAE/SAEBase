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


def test_workplan_rejects_unsafe_step_ids(tmp_path):
    plan = _fixture(tmp_path)
    target = tmp_path / plan
    original = target.read_text(encoding="utf-8")
    for unsafe in ("../escape", "/absolute", "run/id", ".hidden"):
        target.write_text(
            original.replace('"id": "setup"', '"id": ' + json.dumps(unsafe)),
            encoding="utf-8",
        )
        with pytest.raises(life.PlanError, match="Invalid task DAG"):
            life.check(plan, root=tmp_path, check_contracts=False)


def test_dispatch_permissions_are_source_bound(tmp_path):
    from harness.github.work_plan.dispatch import approved_command, plan_commands

    plan_path = _fixture(tmp_path)
    life.check(plan_path, root=tmp_path, check_contracts=False)
    source = tmp_path / "scripts/experiments/example.sh"
    source.parent.mkdir(parents=True)
    source.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    plan = {
        "steps": [
            {
                "id": "setup",
                "needs": [],
                "permission": "local",
                "command": ["python", "-m", "saebase.setup", "--help"],
            },
            {
                "id": "run",
                "needs": ["setup"],
                "permissions": ["network", "gpu"],
                "command": ["bash", "scripts/experiments/example.sh", "--allow-local-gpu"],
            },
        ]
    }
    steps = plan_commands(plan, tmp_path, {"local", "network"})
    assert steps[0]["status"] == "PLANNED"
    assert steps[1]["status"] == "NOT_AUTHORIZED"
    assert plan_commands(plan, tmp_path, {"local", "network", "gpu"})[1]["status"] == "PLANNED"
    with pytest.raises(life.PlanError, match="lacks declared permission"):
        approved_command(
            {
                "permission": "local",
                "command": ["bash", "scripts/experiments/example.sh", "--allow-local-gpu"],
            },
            tmp_path,
        )


def test_dispatch_rejects_implicit_network_and_gpu_execution(tmp_path):
    from harness.github.work_plan.dispatch import approved_command

    for argv in (
        ["python", "-m", "tasks.precheck.worker", "--allow-network"],
        ["python", "-m", "tasks.precheck.worker", "--allow-gpu"],
        ["bash", "scripts/experiments/example.sh", "--execute"],
    ):
        if argv[0] == "bash":
            source = tmp_path / "scripts/experiments/example.sh"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        with pytest.raises(life.PlanError, match="lacks declared permission"):
            approved_command({"permission": "local", "command": argv}, tmp_path)


def test_cross_workplan_requires_exact_human_approval(tmp_path):
    plan = _fixture(tmp_path)
    path = tmp_path / plan
    text = path.read_text(encoding="utf-8")
    path.write_text(text + "\nPrior WP22 is required for this run.\n", encoding="utf-8")
    # Previously reviewed legacy plans remain readable; a new draft is gated.
    assert life.check(plan, root=tmp_path, check_contracts=False)["plan_id"] == "WP23"
    path.write_text(path.read_text().replace('"status": "reviewed"', '"status": "draft"'))
    with pytest.raises(life.PlanError, match="Unapproved cross-WP"):
        life.check(plan, root=tmp_path, check_contracts=False)
    human = tmp_path / "harness/github/work_plan/from_human/WP23.md"
    human.write_text(human.read_text().replace(
        '"scientific_motivation": "Evidence for the research question"',
        '"scientific_motivation": "Evidence for the research question", "cross_workplan_approvals": [{"wp_id": "WP22", "purpose": "historical record reference", "decision": "APPROVED"}]'
    ))
    assert life.check(plan, root=tmp_path, check_contracts=False)["plan_id"] == "WP23"


def test_agent_ready_requires_complete_prepared_evidence(tmp_path):
    plan = _fixture(tmp_path)
    path = tmp_path / plan
    data = path.read_text()
    data = data.replace('"status": "reviewed"', '"status": "agent_ready", "plan_role": "local_agent"')
    path.write_text(data)
    with pytest.raises(life.PlanError, match="actual commands"):
        life.check(plan, root=tmp_path, check_contracts=False)
    data = data.replace('"goal": "Observe setup"', '"command": ["python", "-m", "saebase.setup", "--help"], "goal": "Observe setup"').replace('"goal": "Run frozen benchmark"', '"command": ["python", "-m", "saebase.runtime", "--help"], "goal": "Run frozen benchmark"')
    path.write_text(data)
    with pytest.raises(life.PlanError, match="source SHA"):
        life.check(plan, root=tmp_path, check_contracts=False)
    data = data.replace('"plan_role": "local_agent"', '"plan_role": "local_agent", "readiness": {"source_sha": "' + "a" * 40 + '", "offline_evidence": "offline.json", "integration_evidence": "integration.json", "ci_evidence": "ci.json"}')
    path.write_text(data)
    assert life.check(plan, root=tmp_path, check_contracts=False)["status"] == "agent_ready"
