"""Offline P2 tests use artificial numeric fixtures bound to real site IDs.

No fixture is shipped as a published feature, and no pretrained weight or
network request is used in these regression tests.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import core, explore  # noqa: E402
from backend.auto_interp import propose_interpretation  # noqa: E402
from backend.feature_store import create_feature_store  # noqa: E402
from backend.api import app  # noqa: E402

client = TestClient(app)


@pytest.fixture
def site():
    return next(s for s in core.load_sites() if core.has_complete_identity(s))


def make_trace(site, number: int, *, text: str = "") -> dict:
    text = text or f"A recorded engineering input {number}."
    group = f"engineering:group-{number}"
    return {
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "source_type": "real_checkpoint",
        "sample_id": f"engineering:sample-{number}",
        "prompt_group_id": group,
        "split": "discovery" if number < 3 else "evaluation",
        "prompt_text": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tokenizer_sha256": "a" * 64,
        "model_processing": "source-frozen-test-protocol",
        "complete_sparse_codes": True,
        "public_text_reviewed": True,
        "rights_reviewed": True,
        "tokens": [
            {"position": 0, "token_id": 2, "text": "<bos>", "features": []},
            {
                "position": 1,
                "token_id": 400 + number,
                "text": "Ġnumber",
                "features": [
                    {"feature_id": 10, "activation": float(number)},
                    {"feature_id": 11, "activation": float(5 - number)},
                ],
            },
        ],
    }


def make_label(site):
    return {
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "source_type": "real_checkpoint",
        "feature_id": 10,
        "origin": "model_candidate",
        "description": "Candidate recognition of numerical expressions",
        "author_or_model": "documented-candidate",
        "method": "approved_import",
        "status": "heldout_tested",
        "discovery_group_ids": ["engineering:group-1"],
        "evaluation_group_ids": ["engineering:group-3", "engineering:group-4"],
        "heldout_score": 0.75,
        "approved": True,
    }


def test_exploration_rejects_unreviewed_and_misidentified_input(tmp_path, site):
    item = make_trace(site, 1)
    with pytest.raises(PermissionError):
        explore.publish_exploration([item], [], output=tmp_path / "explore")
    with pytest.raises(PermissionError):
        explore.publish_exploration(
            [{**item, "public_text_reviewed": False}],
            [],
            output=tmp_path / "explore",
            public_consent=True,
        )
    with pytest.raises(ValueError, match="source site"):
        explore.publish_exploration(
            [{**item, "sae_sha256": "0" * 64}],
            [],
            output=tmp_path / "explore",
            public_consent=True,
        )
    with pytest.raises(ValueError, match="complete nonzero"):
        explore.publish_exploration(
            [{**item, "complete_sparse_codes": False}],
            [],
            output=tmp_path / "explore",
            public_consent=True,
        )


def test_exploration_immutable_artifact_and_checksum(tmp_path, site):
    traces = [make_trace(site, i) for i in range(1, 5)]
    meta = explore.publish_exploration(
        traces,
        [make_label(site)],
        output=tmp_path / "explore",
        public_consent=True,
    )
    assert meta["trace_count"] == 4
    assert meta["explanation_count"] == 1
    assert len(explore.published_exploration(tmp_path)["traces"]) == 4
    with pytest.raises(FileExistsError):
        explore.publish_exploration(
            traces,
            [],
            output=tmp_path / "explore",
            public_consent=True,
        )
    file_path = tmp_path / "explore" / "records.json"
    file_path.write_text(file_path.read_text(encoding="utf-8") + " ")
    with pytest.raises(core.InvalidEvidence, match="integrity"):
        explore.published_exploration(tmp_path)


def test_three_search_modes_and_independent_group_correlation(tmp_path, monkeypatch, site):
    root = tmp_path
    obs = []
    for fid in (10, 11):
        for n in range(1, 5):
            obs.append(
                {
                    "site_id": site["site_id"],
                    "model_sha256": site["model_sha256"],
                    "sae_sha256": site["sae_sha256"],
                    "source_type": "real_checkpoint",
                    "feature_id": fid,
                    "token_position": 1,
                    "activation": float(n),
                    "sample_id": f"engineering:sample-{n}",
                    "prompt_group_id": f"engineering:group-{n}",
                    "token_text": "<script>danger</script>" if fid == 10 else "token",
                    "text_excerpt": "Safe example is visibly escaped <script>alert(1)</script>",
                    "public_text_reviewed": True,
                }
            )
    create_feature_store(obs, root / "features")
    explore.publish_exploration(
        [make_trace(site, n) for n in range(1, 5)],
        [make_label(site)],
        output=root / "explore",
        public_consent=True,
    )
    monkeypatch.setenv("OPENSAE_LAB_DATA", str(root))
    listed = client.get("/api/features?q=:F10")
    assert listed.status_code == 200 and listed.json()["total"] == 1
    assert listed.json()["source"] == "verified_sqlite"
    searched = client.post(
        "/api/explore/search",
        json={
            "mode": "explanation",
            "query": "numerical",
        },
    )
    assert searched.status_code == 200
    assert searched.json()["total"] == 1
    assert searched.json()["items"][0]["feature_id"] == 10
    details = client.get(f"/api/explore/features/{site['site_id']}/10")
    assert details.status_code == 200
    assert details.json()["next_feature_id"] == 11
    assert details.json()["previous_feature_id"] is None
    assert details.json()["distribution"]["population_density"] is None
    assert details.json()["site"]["model_sha256"] == site["model_sha256"]
    export = client.get(f"/api/explore/features/{site['site_id']}/10/export.html")
    assert export.status_code == 200
    assert "<script>alert(1)</script>" not in export.text
    assert "&lt;script&gt;" in export.text
    assert "no external" not in export.text.lower()
    summaries = client.get("/api/explore/traces")
    assert summaries.status_code == 200 and summaries.json()["total"] == 4
    tid = summaries.json()["items"][0]["trace_id"]
    saved = client.get("/api/explore/traces/" + tid)
    assert saved.status_code == 200 and saved.json()["trace_id"] == tid
    by_input = client.post(
        "/api/explore/search",
        json={
            "mode": "inference",
            "query": make_trace(site, 1)["prompt_text"],
        },
    )
    assert by_input.status_code == 200
    assert by_input.json()["mode"] == "recorded_only"
    assert by_input.json()["items"][0]["max_features"][0]["feature_id"] == 11
    by_token = client.post(
        "/api/explore/search",
        json={
            "mode": "token",
            "trace_id": tid,
            "position": 1,
        },
    )
    assert by_token.status_code == 200
    assert len(by_token.json()["items"]) == 2
    assert by_token.json()["items"][0]["position"] == 1
    rho = client.get(f"/api/explore/features/{site['site_id']}/10/correlations")
    assert rho.status_code == 200
    assert rho.json()["n_groups"] == 4
    assert rho.json()["items"][0]["feature_id"] == 11
    assert rho.json()["items"][0]["pearson_r"] == pytest.approx(-1.0)
    unseen = client.post(
        "/api/explore/search",
        json={
            "mode": "inference",
            "query": "A prompt never sent through a model.",
        },
    )
    assert unseen.status_code == 503
    assert client.get("/api/explore/traces/" + "0" * 64).status_code == 404
    assert (
        client.post(
            "/api/explore/search",
            json={
                "mode": "token",
                "trace_id": tid,
                "position": 10,
            },
        ).status_code
        == 422
    )


def test_interpretation_budget_splits_and_provenance():
    calls = []
    discovery = [{"group_id": "d", "text": "Positive discovery context", "activation": 3.5}]
    heldout = [
        {"group_id": "e1", "text": "Positive heldout", "activation": 2},
        {"group_id": "e2", "text": "Negative heldout", "activation": 0},
    ]

    def generate(examples):
        calls.append(list(examples))
        return "A candidate about positive text"

    def predict(_description, text):
        return "Positive" in text

    with pytest.raises(PermissionError):
        propose_interpretation(
            site_id="actual-site",
            feature_id=10,
            discovery=discovery,
            heldout=heldout,
            generator=generate,
            generator_model="external-opt-in",
            generator_method="example-conditioned",
            authorized_budget_usd=0,
            estimated_cost_usd=1,
            predictor=predict,
        )
    assert calls == []
    result = propose_interpretation(
        site_id="actual-site",
        feature_id=10,
        discovery=discovery,
        heldout=heldout,
        generator=generate,
        generator_model="external-opt-in",
        generator_method="example-conditioned",
        authorized_budget_usd=1,
        estimated_cost_usd=0.25,
        predictor=predict,
    )
    assert len(calls) == 1
    assert result["status"] == "heldout_tested"
    assert result["approved"] is False
    assert result["heldout_score"] == 1.0
    assert result["evaluation_kind"].endswith("_not_semantic_truth")
    with pytest.raises(ValueError, match="overlap"):
        propose_interpretation(
            site_id="actual-site",
            feature_id=10,
            discovery=discovery,
            heldout=[{**heldout[0], "group_id": "d"}],
            generator=generate,
            generator_model="test",
            generator_method="method",
            authorized_budget_usd=1,
            estimated_cost_usd=0,
        )
