"""Human Check decisions are independent, pre-PR and never mutate Atlas authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from harness.core import agent_flow
from harness.github import human_check as hc


def _write(root: Path, name: str, content: object) -> None:
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        content if isinstance(content, str) else json.dumps(content),
        encoding="utf-8",
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(agent_flow, "ROOT", tmp_path)
    monkeypatch.setattr(hc, "ROOT", tmp_path)
    monkeypatch.setattr(hc, "WORKSPACE", tmp_path / "experiment/_human_check")
    _write(
        tmp_path,
        "harness/core/project_graph.json",
        {
            "nodes": [{"id": "g.registry", "title": "Registry", "description": "Shared registry"}],
            "edges": [],
        },
    )
    _write(
        tmp_path,
        "doc/knowledge/context_graph.json",
        {
            "nodes": [{"id": "q.sae.methods", "title": "Methods", "description": "Methods"}],
            "edges": [],
        },
    )
    _write(
        tmp_path,
        "doc/code/code_tree.json",
        {
            "nodes": [
                {
                    "id": "code:module:saebase.registry",
                    "title": "Registry",
                    "description": "Shared model and task registry",
                    "path": "saebase/registry.py",
                }
            ],
            "edges": [],
        },
    )
    _write(
        tmp_path,
        "doc/knowledge/glossary.json",
        {
            "terms": [
                {
                    "id": "term.sae.reproducible",
                    "term": "Reproducible",
                    "definition": "Repeatable evaluations",
                }
            ],
        },
    )
    _write(
        tmp_path,
        "doc/knowledge/atlas_terms.json",
        {
            "terms": [
                {"id": "term.registry", "term": "Registry", "definition": "The registered keys"}
            ],
        },
    )
    _write(tmp_path, "saebase/registry.py", "ORIGINAL = 1\n")
    _write(tmp_path, "tasks/capture/operation.py", "ORIGINAL = 2\n")
    _write(tmp_path, "saebase/tools/new.py", "ORIGINAL = 3\n")
    return tmp_path


def _manifest():
    return {
        "id": "HC-Example-001",
        "goal": "Independently review registry and capture changes",
        "items": [
            {
                "view": "core",
                "node_id": "g.registry",
                "operation": "change",
                "after": {"title": "Registry", "description": "Clarify public behavior"},
                "files": ["saebase/registry.py"],
                "reason": "Existing Core architecture",
                "context_term_ids": ["term.registry"],
                "checks": ["Registry regression"],
            },
            {
                "view": "code",
                "node_id": "code:module:saebase.registry",
                "operation": "change",
                "after": {"title": "Registry", "description": "Change its implementation"},
                "files": ["saebase/registry.py"],
                "reason": "Actual source behavior",
                "context_term_ids": ["term.registry"],
            },
            {
                "view": "knowledge",
                "node_id": "q.sae.methods",
                "operation": "change",
                "after": {"title": "Methods", "description": "Improve repeatability notes"},
                "files": ["tasks/capture/operation.py"],
                "reason": "Existing Knowledge",
                "context_term_ids": ["term.sae.reproducible"],
            },
        ],
    }


def test_node_decisions_are_separate_and_authority_is_unchanged(workspace):
    before = {
        p: (workspace / p).read_bytes()
        for p in (
            "harness/core/project_graph.json",
            "doc/knowledge/context_graph.json",
            "doc/code/code_tree.json",
            "doc/knowledge/glossary.json",
        )
    }
    plan = hc.create(_manifest(), base_sha="a" * 40)
    assert plan["mode"] == "on"
    assert "Before:" in hc.report_markdown(plan)
    with pytest.raises(agent_flow.FlowError, match="Unapproved"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="a" * 40)
    hc.decide(plan, "knowledge:q.sae.methods", "approved", actor="Human")
    hc.decide(plan, "core:g.registry", "denied", actor="Human")
    assert (
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="a" * 40)["status"]
        == "APPROVED_SCOPE"
    )
    with pytest.raises(agent_flow.FlowError, match="Unapproved"):
        hc.allowed_paths(plan, ["saebase/registry.py"], base_sha="a" * 40)
    assert hc.load(plan["id"])["revision"] == 2
    for path, content in before.items():
        assert (workspace / path).read_bytes() == content


def test_repeated_decisions_reset_and_stale_sources_fail(workspace):
    plan = hc.create(_manifest(), base_sha="b" * 40)
    hc.decide(plan, "knowledge:q.sae.methods", "approved", actor="Human")
    hc.decide(plan, "knowledge:q.sae.methods", "unchecked", actor="Human")
    with pytest.raises(agent_flow.FlowError, match="Unapproved"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="b" * 40)
    hc.decide(plan, "knowledge:q.sae.methods", "approved", actor="Human")
    with pytest.raises(agent_flow.FlowError, match="commit drift"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="c" * 40)
    (workspace / "tasks/capture/operation.py").write_text("CHANGED = 1\n", encoding="utf-8")
    with pytest.raises(agent_flow.FlowError, match="Source changed"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="b" * 40)
    assert (
        hc.allowed_paths(
            plan,
            ["tasks/capture/operation.py"],
            base_sha="b" * 40,
            before_edit=False,
        )["status"]
        == "APPROVED_SCOPE"
    )


def test_human_opt_out_is_explicit_and_independently_recorded(workspace):
    plan = hc.create(_manifest(), base_sha="c" * 40)
    with pytest.raises(agent_flow.FlowError, match="Unapproved"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="c" * 40)
    hc.set_mode(plan, False, actor="Human")
    assert hc.load(plan["id"])["opt_out"]["actor"] == "Human"
    assert (
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="c" * 40)["status"]
        == "HUMAN_CHECK_OPTED_OUT"
    )
    with pytest.raises(agent_flow.FlowError, match="disabled"):
        hc.decide(plan, "knowledge:q.sae.methods", "approved", actor="Human")
    hc.set_mode(plan, True, actor="Human")
    with pytest.raises(agent_flow.FlowError, match="Unapproved"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="c" * 40)


def test_mixed_decisions_on_same_path_fail_closed_not_other_paths(workspace):
    plan = hc.create(_manifest(), base_sha="d" * 40)
    hc.decide(plan, "core:g.registry", "approved", actor="Human")
    hc.decide(plan, "code:code:module:saebase.registry", "denied", actor="Human")
    hc.decide(plan, "knowledge:q.sae.methods", "approved", actor="Human")
    with pytest.raises(agent_flow.FlowError, match="overlapping denied"):
        hc.allowed_paths(plan, ["saebase/registry.py"], base_sha="d" * 40)
    assert (
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="d" * 40)["status"]
        == "APPROVED_SCOPE"
    )


def test_proposal_is_sealed_and_absent_nodes_are_not_implemented(workspace):
    plan = hc.create(_manifest(), base_sha="e" * 40)
    tampered = copy.deepcopy(plan)
    tampered["items"][0]["after"]["description"] = "An unauthorized plan rewrite"
    with pytest.raises(agent_flow.FlowError, match="changed"):
        hc.validate(tampered)
    manifest = _manifest()
    manifest["id"] = "HC-Example-002"
    manifest["items"] = [
        {
            "view": "code",
            "node_id": "code:module:saebase.new_feature",
            "operation": "add",
            "after": {
                "title": "Proposed feature",
                "description": "Pending actual Python AST",
                "path": "saebase/tools/new.py",
            },
            "files": ["saebase/tools/new.py"],
            "reason": "Optional proposed code node",
            "context_term_ids": ["term.registry"],
        }
    ]
    proposed = hc.create(manifest, base_sha="f" * 40)
    assert proposed["items"][0]["before"] is None
    assert (
        not (workspace / "doc/code/code_tree.json")
        .read_text()
        .find("code:module:saebase.new_feature")
        >= 0
    )


def test_loopback_atlas_persists_single_node_and_opt_out(workspace):
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from harness.core.agent_review_server import create_server

    hc.create(_manifest(), base_sha="a" * 40)
    server = create_server("HC-Example-001", 0, human_check=True)
    runner = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    origin = "http://127.0.0.1:" + str(server.server_address[1])

    def fetch(path):
        with urlopen(origin + path, timeout=5) as response:
            return response.read().decode("utf-8")

    def post(path, body, *, key=None):
        payload = json.dumps(body).encode("utf-8")
        req = Request(
            origin + path,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "X-Atlas-Review-Key": key or "",
            },
            method="POST",
        )
        with urlopen(req, timeout=5) as response:
            return json.loads(response.read())

    try:
        assert "Human Check" in fetch("/api/review/report")
        state = json.loads(fetch("/api/review/state"))
        assert state["plan"]["mode"] == "on"
        with pytest.raises(HTTPError) as forbidden:
            post(
                "/api/review/decision",
                {
                    "revision": 0,
                    "key": "knowledge:q.sae.methods",
                    "decision": "approved",
                },
            )
        assert forbidden.value.code == 403
        approved = post(
            "/api/review/decision",
            {
                "revision": 0,
                "key": "knowledge:q.sae.methods",
                "decision": "approved",
            },
            key=server.review_key,
        )
        assert approved["plan"]["items"][2]["proposal_review"]["decision"] == "approved"
        assert approved["plan"]["items"][0]["proposal_review"] is None
        switched = post(
            "/api/review/mode",
            {
                "revision": 1,
                "enabled": False,
            },
            key=server.review_key,
        )
        assert switched["plan"]["opt_out"]["actor"] == "Human via local Atlas"
        assert hc.load("HC-Example-001")["mode"] == "off"
        with pytest.raises(HTTPError) as stale:
            post(
                "/api/review/mode",
                {
                    "revision": 1,
                    "enabled": True,
                },
                key=server.review_key,
            )
        assert stale.value.code == 409
        assert "Registry" in fetch("/api/review/report")
    finally:
        server.shutdown()
        server.server_close()
        runner.join(timeout=5)


def test_authority_drift_invalidates_pre_edit_approval(workspace):
    plan = hc.create(_manifest(), base_sha="f" * 40)
    hc.decide(plan, "knowledge:q.sae.methods", "approved", actor="Human")
    path = workspace / "harness/core/project_graph.json"
    path.write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
    with pytest.raises(agent_flow.FlowError, match="Atlas authority drift"):
        hc.allowed_paths(plan, ["tasks/capture/operation.py"], base_sha="f" * 40)


def test_logical_group_requires_visiting_every_atlas_part(workspace):
    manifest = _manifest()
    manifest["id"] = "HC-Group-001"
    for item in manifest["items"][:2]:
        item["group_id"] = "HC01"
        item["group_title"] = "One change across Core and Code"
    item = manifest["items"][2]
    item["group_id"] = "HC02"
    item["group_title"] = "Independent Knowledge change"
    plan = hc.create(manifest, base_sha="a" * 40)
    assert len(plan["groups"]) == 2
    with pytest.raises(agent_flow.FlowError, match="Grouped Atlas part"):
        hc.decide(plan, "core:g.registry", "approved", actor="Human")
    hc.visit(plan, "core:g.registry", actor="Human")
    with pytest.raises(agent_flow.FlowError, match="Visit every linked Atlas part"):
        hc.decide_group(plan, "HC01", "approved", actor="Human")
    hc.visit(plan, "code:code:module:saebase.registry", actor="Human")
    hc.decide_group(plan, "HC01", "denied", actor="Human")
    hc.visit(plan, "knowledge:q.sae.methods", actor="Human")
    hc.decide_group(plan, "HC02", "approved", actor="Human")
    assert hc.load(plan["id"])["revision"] == 5
    assert hc.allowed_paths(
        plan, ["tasks/capture/operation.py"], base_sha="a" * 40
    )["status"] == "APPROVED_SCOPE"
    with pytest.raises(agent_flow.FlowError, match="Unapproved"):
        hc.allowed_paths(plan, ["saebase/registry.py"], base_sha="a" * 40)


def test_review_server_group_endpoints_check_nonce_and_revision(workspace):
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from harness.core.agent_review_server import create_server

    manifest = _manifest()
    manifest["id"] = "HC-Group-002"
    for item in manifest["items"]:
        item["group_id"] = "HC01"
        item["group_title"] = "Three linked Atlas parts"
    hc.create(manifest, base_sha="b" * 40)
    server = create_server("HC-Group-002", 0, human_check=True)
    runner = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    origin = "http://127.0.0.1:" + str(server.server_address[1])

    def post(path, body, key=None):
        request = Request(
            origin + path, data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "X-Atlas-Review-Key": key or "",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    try:
        with pytest.raises(HTTPError) as error:
            post("/api/review/visit", {
                "revision": 0, "key": "core:g.registry"
            })
        assert error.value.code == 403
        for revision, node in enumerate([
            "core:g.registry",
            "code:code:module:saebase.registry",
            "knowledge:q.sae.methods",
        ]):
            result = post("/api/review/visit", {
                "revision": revision, "key": node,
            }, key=server.review_key)
            assert result["plan"]["revision"] == revision + 1
        final = post("/api/review/group-decision", {
            "revision": 3, "group_id": "HC01", "decision": "approved",
        }, key=server.review_key)
        assert all(i["proposal_review"]["decision"] == "approved"
                   for i in final["plan"]["items"])
        with pytest.raises(HTTPError) as stale:
            post("/api/review/group-decision", {
                "revision": 3, "group_id": "HC01", "decision": "denied",
            }, key=server.review_key)
        assert stale.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        runner.join(timeout=5)


def test_atlas_group_review_frontend_opens_core_and_guides_cross_tree_review():
    from pathlib import Path

    app = (Path(__file__).resolve().parents[2] / "visualization/app.js").read_text()
    assert "if(requestedReview())" in app
    assert "function logicalReviewGroup(" in app
    assert "/api/review/visit" in app
    assert "/api/review/group-decision" in app
    assert "Next linked part" not in app
    assert "atlasReviewGroupScope" not in app
    assert "maybeShowReviewDone" in app
    assert "reviewQueue=[...ordered,...byKey.values()]" in app
    assert "groupComplete" in app


def test_quickstart_reuses_proposal_and_checks_ui(workspace, monkeypatch, tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit
    from harness.core import agent_review_server

    class FakeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/review/state":
                body = json.dumps({"plan": {"id": "HC-Example-001"}}).encode()
            elif path == "/core.html":
                body = b"<html><script src='app.js'></script></html>"
            else:
                body = b"static asset " * 20
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass

    def fake_server(plan_id, port, *, human_check):
        assert human_check and plan_id == "HC-Example-001"
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        server.review_key = "test-only"
        return server

    monkeypatch.setattr(agent_review_server, "create_server", fake_server)
    monkeypatch.setattr(hc, "_git_head", lambda: "a" * 40)
    # Avoid the interactive serve-forever wait while retaining the HTTP probe.
    original_join = threading.Thread.join
    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: None)
    manifest = tmp_path / "proposal.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    hc.quickstart_review(manifest, base_sha="a" * 40, port=0, no_browser=True)
    assert hc.load("HC-Example-001")["base_sha"] == "a" * 40
