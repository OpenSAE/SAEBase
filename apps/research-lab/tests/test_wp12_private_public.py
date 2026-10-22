"""Fast CPU public/private WP12 integration and privacy checks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import core, published_wp12  # noqa: E402
from backend.api import app  # noqa: E402

client = TestClient(app)


def graph_record() -> dict:
    return {
        "schema": "opensae.wp12.p6.graph.v1",
        "nodes": [
            {"id": "token", "kind": "token", "label": "A"},
            {"id": "f", "kind": "feature", "label": "Feature 4", "feature_id": 4},
            {"id": "error", "kind": "error", "label": "Residual error"},
            {"id": "logit", "kind": "logit", "label": "Target"},
        ],
        "edges": [
            {"source": "token", "target": "f", "weight": -0.2},
            {"source": "f", "target": "logit", "weight": 0.4},
            {"source": "error", "target": "logit", "weight": 0.02},
        ],
        "provenance": {"source_sha256": "a" * 64, "original_tool_commit": "b" * 40},
    }


def test_only_reviewed_native_records_become_browsable(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENSAE_LAB_DATA", str(tmp_path))
    assert client.get("/api/wp12/circuits").json()["total"] == 0
    with pytest.raises(PermissionError):
        published_wp12.publish_record(
            "circuits",
            graph_record(),
            root=tmp_path,
        )
    record = published_wp12.publish_record(
        "circuits",
        graph_record(),
        root=tmp_path,
        public_consent=True,
        rights_reviewed=True,
    )
    sha = record["record_sha256"]
    response = client.get("/api/wp12/circuits")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    detail = client.get("/api/wp12/circuits/" + sha)
    assert detail.status_code == 200
    assert detail.json()["edges"][0]["weight"] == -0.2
    assert client.get("/api/wp12/circuits/" + "f" * 64).status_code == 404
    assert client.get("/api/wp12/invalid").status_code == 404
    path = tmp_path / "published_wp12" / "circuits" / sha / "record.json"
    path.write_text(path.read_text() + "\n")
    assert client.get("/api/wp12/circuits/" + sha).status_code == 503


def test_public_worker_stays_disabled_even_with_input(monkeypatch):
    monkeypatch.delenv("OPENSAE_ENABLE_PRIVATE_JOBS", raising=False)
    response = client.post("/api/private/inputs", json={"text": "A private prompt."})
    assert response.status_code == 503
    assert client.get("/api/wp12-compute").json()["online_new_inference"] is False


def test_private_operator_worker_roles_and_durable_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENSAE_ENABLE_PRIVATE_JOBS", "1")
    monkeypatch.setenv("OPENSAE_PRIVATE_DATA", str(tmp_path))
    monkeypatch.setenv("OPENSAE_PRIVATE_OPERATOR_TOKEN", "a" * 40)
    monkeypatch.setenv("OPENSAE_PRIVATE_WORKER_TOKEN", "b" * 40)
    op = {"Authorization": "Bearer " + "a" * 40}
    worker = {"Authorization": "Bearer " + "b" * 40}
    assert (
        client.post("/api/private/inputs", json={"text": "Hidden"}, headers=worker).status_code
        == 401
    )
    r = client.post("/api/private/inputs", json={"text": "Hidden"}, headers=op)
    assert r.status_code == 200
    ref = r.json()["input_ref"]
    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    spec = {
        "client_key": "wp12-test-001",
        "site_id": site["site_id"],
        "operation": "ablate",
        "feature_ids": [3],
        "strength": 1.0,
        "max_new_tokens": 0,
        "input_ref": ref,
    }
    denied = client.post("/api/private/jobs", json=spec, headers=worker)
    assert denied.status_code == 401
    submitted = client.post("/api/private/jobs", json=spec, headers=op)
    assert submitted.status_code == 200
    job_id = submitted.json()["job_id"]
    assert client.post("/api/private/jobs", json=spec, headers=op).json()["deduplicated"]
    claimed = client.post("/api/private/workers/claim", headers=worker)
    assert claimed.status_code == 200
    job = claimed.json()["job"]
    assert "Hidden" not in str(job)
    payload = {
        "job_id": job_id,
        "schema": "opensae.wp12.private_test_fixture.v1",
        "logit_delta_l2": 0.25,
    }
    data = json.dumps(payload, sort_keys=True).encode()
    path = tmp_path / "results" / (job_id + ".json")
    path.parent.mkdir()
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    finished = client.post(
        "/api/private/workers/finish",
        headers=worker,
        json={
            "job_id": job_id,
            "worker_id": "opensaebase-worker",
            "attempt": job["attempt"],
            "result_sha256": sha,
        },
    )
    assert finished.status_code == 200 and finished.json()["accepted"]
    status = client.get("/api/private/jobs/" + job_id, headers=op)
    assert status.json()["status"] == "succeeded"
    assert "Hidden" not in status.text
    assert client.get("/api/private/jobs/" + job_id + "/result", headers=worker).status_code == 401
    checked = client.get("/api/private/jobs/" + job_id + "/result", headers=op)
    assert checked.status_code == 200
    assert checked.json()["logit_delta_l2"] == 0.25
    path.write_bytes(b'{"job_id":"modified"}')
    assert client.get("/api/private/jobs/" + job_id + "/result", headers=op).status_code == 503
