"""Negative source-access tests: readable is not inferred from file existence."""

import json
from pathlib import Path

from harness.knowledge.dialogue_context import read_project_context


def _fixture(tmp_path: Path):
    p = tmp_path / "doc/knowledge/context_graph.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"schema_version": 1, "nodes": [{"id": "q.one", "title": "Visible"}]}))
    code = tmp_path / "src/mod.py"
    code.parent.mkdir()
    code.write_text("def f():\n    return 1\n")
    env = tmp_path / "harness/environment/example.json"
    env.parent.mkdir(parents=True)
    env.write_text(
        json.dumps({"synthetic": True, "captured_at_utc": None, "gpu": {"status": "synthetic"}})
    )
    return p, code, env


def test_allowlisted_knowledge_and_exact_code(tmp_path):
    p, code, _ = _fixture(tmp_path)
    r = read_project_context(
        tmp_path,
        [
            {"kind": "knowledge", "id": "q.one"},
            {"kind": "code", "id": "src/mod.py"},
            {"kind": "knowledge", "id": "q.missing"},
        ],
        [str(p.relative_to(tmp_path)), str(code.relative_to(tmp_path))],
    )
    assert [x["status"] for x in r["results"]] == ["READ", "READ", "UNRESOLVED_ID"]
    assert r["results"][1]["value"]["functions"] == ["f"]
    assert len(r["results"][0]["source_sha256"]) == 64


def test_unapproved_existing_source_is_blocked(tmp_path):
    _fixture(tmp_path)
    r = read_project_context(tmp_path, [{"kind": "knowledge", "id": "q.one"}], [])
    assert r["results"][0]["status"] == "BLOCKED_SCOPE"
    assert "value" not in r["results"][0]


def test_escape_and_symlink_are_unresolved(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "src/escape.py").symlink_to(tmp_path.parent / "missing.py")
    r = read_project_context(
        tmp_path,
        [
            {"kind": "code", "id": "../other.py"},
            {"kind": "code", "id": "src/escape.py"},
        ],
        ["../other.py", "src/escape.py"],
    )
    assert [x["status"] for x in r["results"]] == ["UNRESOLVED", "UNRESOLVED"]


def test_environment_example_is_never_current_machine(tmp_path):
    _, _, env = _fixture(tmp_path)
    r = read_project_context(
        tmp_path,
        [{"kind": "environment", "id": "harness/environment/example.json"}],
        [str(env.relative_to(tmp_path))],
    )
    assert r["results"][0]["status"] == "SYNTHETIC"


def test_missing_file_unknown_not_verified(tmp_path):
    r = read_project_context(tmp_path, [{"kind": "code", "id": "src/no.py"}], ["src/no.py"])
    assert r["results"][0]["status"] == "UNRESOLVED"


def test_no_default_scanning(tmp_path):
    _fixture(tmp_path)
    assert read_project_context(tmp_path, [], [])["results"] == []
