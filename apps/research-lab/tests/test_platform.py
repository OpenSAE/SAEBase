"""No GPU, network, pretrained weights or synthetic-as-real results required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend import core
from backend.feature_store import create_feature_store,query_feature_store,feature_store_count  # noqa: E402
from backend.ingest import build_feature_index, freeze_dataset, normalize, select_records  # noqa: E402
from backend.publish import publish_report  # noqa: E402
from backend.api import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


@pytest.fixture
def raw_rows():
    return [
        {
            "id": str(i),
            "text": "Once there was a different story number " + str(i) + ".",
            "unused": "x",
        }
        for i in range(70)
    ]


def test_actual_repository_evidence_is_unchanged_and_joinable():
    sites = core.load_sites()
    assert len(sites) == 48
    assert len({s["site_id"] for s in sites}) == 48
    assert all(s["status"] == "verified_engineering" and s["group_count"] == 64 for s in sites)
    assert all(s["evidence_url"].startswith("https://github.com/OpenSAE/") for s in sites)
    # Exactly three older CPU acceptances have no stored full-weight digest.
    # Historical evidence can still be displayed, but not used to certify a new run.
    assert sum(core.has_complete_identity(s) for s in sites) == 45
    metrics = core.historical_metrics(sites)
    assert len(metrics) >= 48 * 4
    assert all(
        row["data_class"] == "historical_engineering" and row["n_valid"] == 64 for row in metrics
    )


def test_data_registry_has_ten_unique_50_target_sources():
    cat = core.dataset_catalog()
    assert len(cat["sources"]) == 10
    assert len({x["id"] for x in cat["sources"]}) == 10
    assert all(x["target"] == 50 and x["license"] == "review_required" for x in cat["sources"])


def test_supported_input_renderers_do_not_leak_gold_answers():
    fixtures = {
        "wikitext103": {
            "article_id": "article-1",
            "id": "w1",
            "text": "A fictional encyclopedia paragraph.",
        },
        "tinystories": {"id": "ts1", "text": "Once upon a time the children went outside."},
        "squad": {
            "id": "s1",
            "title": "Source article",
            "context": "The sky is blue.",
            "question": "What color is the sky?",
            "answers": {"text": ["blue"], "answer_start": [11]},
        },
        "boolq": {
            "id": "b1",
            "passage": "The Nile is a river.",
            "question": "Is the Nile a river?",
            "answer": True,
        },
        "arc_challenge": {
            "id": "a1",
            "question": "Which is solid?",
            "choices": {"label": ["A", "B"], "text": ["rock", "steam"]},
            "answerKey": "A",
        },
        "hellaswag": {
            "id": "h1",
            "ctx": "The person opened the door.",
            "endings": ["They left.", "They slept.", "They laughed.", "They closed it."],
            "label": "2",
        },
        "winogrande": {
            "qID": "w1",
            "sentence": "The elephant was larger than the _.",
            "option1": "mouse",
            "option2": "tree",
            "answer": "1",
        },
        "gsm8k": {"question": "What is two plus two?", "answer": "#### 4"},
        "humaneval": {
            "task_id": "HumanEval/1",
            "prompt": "def test(x):\n    pass\n",
            "canonical_solution": "return x + 1",
            "test": "assert test(1) == 2",
        },
        "truthfulqa": {
            "question": "Is a rock alive?",
            "mc1_targets": {"choices": ["Yes", "No"], "labels": [0, 1]},
        },
    }
    for source, row in fixtures.items():
        output = normalize(source, row)
        assert output["prompt"].strip()
        assert output["source_group_id"]
        assert output["source_row_id"]
        if source == "squad":
            assert "blue." in output["prompt"]  # gold in provided context is not leakage
        if source == "gsm8k":
            assert "#### 4" not in output["prompt"]
        if source == "humaneval":
            assert "canonical_solution" not in output["prompt"]
        if source == "boolq":
            assert output["task_label"] == "yes"
    assert "option" not in normalize("winogrande", fixtures["winogrande"])["prompt"].lower()


def test_deterministic_sample_and_split(raw_rows):
    picked = select_records("tinystories", raw_rows)
    rotated = list(reversed(raw_rows))
    assert picked == select_records("tinystories", rotated)
    assert len(picked) == 50
    assert [x["usage_split"] for x in picked].count("discovery") == 20
    assert [x["usage_split"] for x in picked].count("evaluation") == 30
    assert len({x["prompt_group_id"] for x in picked}) == 50
    assert len({x["sample_id"] for x in picked}) == 50


def test_boolq_gold_label_balance():
    inputs = [
        {"passage": "Unique passage " + str(i), "question": "Is it true?", "answer": (i % 2) == 0}
        for i in range(80)
    ]
    rows = select_records("boolq", inputs)
    assert len(rows) == 50
    assert sum(r["task_label"] == "yes" for r in rows) == 25
    assert sum(r["task_label"] == "no" for r in rows) == 25


def test_immutable_freeze_and_release_visibility(tmp_path, raw_rows):
    output = tmp_path / "datasets" / "tinystories"
    manifest = freeze_dataset(
        "tinystories", raw_rows, source_revision="deadbeef0123456789deadbeef0123456789abcdef", output=output
    )
    assert manifest["sample_count"] == 50
    assert not manifest["human_reviewed"]
    assert core.imported_counts(tmp_path) == {}
    assert core.digest(output / "records.jsonl") == manifest["records_sha256"]
    with pytest.raises(FileExistsError):
        freeze_dataset("tinystories", raw_rows, source_revision="deadbeef0123456789deadbeef0123456789abcdef", output=output)
    reviewed = tmp_path / "datasets" / "boolq"
    inputs = [
        {"passage": "Unique passage " + str(i), "question": "Yes?", "answer": i % 2 == 0}
        for i in range(80)
    ]
    freeze_dataset(
        "boolq",
        inputs,
        source_revision="deadbeef0123456789deadbeef0123456789abcdef",
        output=reviewed,
        license_reviewed=True,
        human_reviewed=True,
    )
    assert core.imported_counts(tmp_path) == {"boolq": 50}
    with (reviewed / "records.jsonl").open("ab") as f:
        f.write(b"tampered")
    assert core.imported_counts(tmp_path) == {}


def test_no_fake_observations_and_integrity(tmp_path):
    site = core.load_sites()[0]
    row = {
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "source_type": "real_checkpoint",
        "feature_id": 123,
        "token_position": 5,
        "activation": 3.25,
        "prompt_group_id": "wiki:article1",
        "sample_id": "wiki:row1",
        "token_text": "planet",
        "text_excerpt": "A planet rotates.",
        "public_text_reviewed": True,
    }
    with pytest.raises(ValueError, match="Synthetic"):
        build_feature_index([{**row, "source_type": "synthetic_demo"}], output=tmp_path / "bad")
    with pytest.raises(ValueError, match="identity"):
        build_feature_index([{**row, "sae_sha256": "bad"}], output=tmp_path / "bad2")
    with pytest.raises(ValueError, match="Nonfinite"):
        build_feature_index([{**row, "activation": float("nan")}], output=tmp_path / "bad3")
    report = build_feature_index(
        [row, {**row, "activation": 2.0, "public_text_reviewed": False}],
        output=tmp_path / "features",
    )
    assert report["feature_count"] == 1
    data = core.feature_index(tmp_path)
    assert data[0]["unique_prompt_groups"] == 1
    assert data[0]["observations"] == 2
    assert len(data[0]["examples"]) == 1
    with (tmp_path / "features" / "index.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(core.InvalidEvidence):
        core.feature_index(tmp_path)


def test_share_and_reproduce_manifest_cannot_be_silently_rewritten(tmp_path):
    record = {
        "visibility": "public",
        "source_type": "real_checkpoint",
        "kind": "sar",
        "protocol_id": "sar.original.v1",
        "source": {"commit": "0123456789abcdef"},
        "title": "Verified SAR run",
        "summary": "Description of evaluated sample groups",
        "results": [
            {"metric": "sar1", "status": "completed", "value": 0.12},
            {"metric": "rho2", "status": "undefined", "value": None},
        ],
    }
    with pytest.raises(PermissionError):
        publish_report(record, output=tmp_path / "reports", public_consent=False)
    created = publish_report(record, output=tmp_path / "reports", public_consent=True)
    report_id = created["report_id"]
    assert core.public_report(report_id, tmp_path)["kind"] == "sar"
    assert len(core.published_reports(tmp_path)) == 1
    with pytest.raises(FileExistsError):
        publish_report(record, output=tmp_path / "reports", public_consent=True)
    with pytest.raises(ValueError):
        publish_report(
            {**record, "results": [{"metric": "oops", "status": "blocked", "value": 99}]},
            output=tmp_path / "oops",
            public_consent=True,
        )
    path = tmp_path / "reports" / report_id / "report.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(core.InvalidEvidence):
        core.public_report(report_id, tmp_path)


def test_readonly_api_and_real_source_status():
    assert client.get("/").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
    info = client.get("/api/summary")
    assert info.status_code == 200
    payload = info.json()
    assert payload["sites_verified"] == 48
    assert payload["gpu_online"] is False
    assert payload["online_new_inference"] is False
    assert payload["historical_groups_per_site"] == 64
    assert client.get("/api/results?metric=core.l0").json()["total"] == 48
    assert (
        client.get("/api/datasets").json()["sources"][0]["status"] == "awaiting_source_and_review"
    )
    assert client.get("/api/features").json()["total"] == 0
    assert client.get("/api/reports/not-a-sha").status_code == 404
    assert client.post("/api/jobs").status_code == 503


def test_preflight_is_not_an_execution_request():
    site = core.load_sites()[0]
    body = {
        "site_id": site["site_id"],
        "text": "A test sentence.",
        "operation": "ablate",
        "feature_ids": [3, 5],
        "max_new_tokens": 16,
    }
    response = client.post("/api/jobs/preflight", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["submitted"] is False and result["ran_model"] is False
    assert result["status"].startswith("blocked_until_gpu")
    assert "text" not in json.dumps(result)
    assert (
        client.post("/api/jobs/preflight", json={**body, "feature_ids": [3, 3]}).status_code == 422
    )
    assert (
        client.post("/api/jobs/preflight", json={**body, "operation": "do_everything"}).status_code
        == 422
    )
    assert (
        client.post("/api/jobs/preflight", json={**body, "site_id": "unverified"}).status_code
        == 422
    )



def test_sqlite_feature_store_indexes_real_records_and_detects_tampering(tmp_path):
    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    sample = {
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "source_type": "real_checkpoint",
        "feature_id": 6,
        "token_position": 2,
        "activation": 1.5,
        "sample_id": "boolq:row1",
        "prompt_group_id": "boolq:group1",
        "public_text_reviewed": True,
        "token_text": "truth",
        "text_excerpt": "A reviewed example about truth.",
    }
    rows = [
        {**sample, "activation": (n + 1) / 10, "sample_id": "boolq:row" + str(n)}
        for n in range(17)
    ]
    rows.extend([{**sample,"feature_id":7,"public_text_reviewed":False,"activation":2}])
    result = create_feature_store(rows, tmp_path / "features")
    assert result["feature_count"] == 2
    assert result["observation_count"] == 18
    assert result["public_examples_stored"] == 8
    assert feature_store_count(tmp_path) == 2
    records = query_feature_store(root=tmp_path,query=":F6",limit=5)
    assert records["total"] == 1
    assert records["items"][0]["observations"] == 17
    assert records["items"][0]["unique_prompt_groups"] == 1
    assert len(records["items"][0]["examples"]) == 8
    # SQL wildcards do not alter exact literal searches.
    assert query_feature_store(root=tmp_path,query="%")["total"] == 0
    assert query_feature_store(root=tmp_path,query="_")["total"] == 0
    with pytest.raises(FileExistsError):
        create_feature_store([sample], tmp_path / "features")
    with (tmp_path / "features" / "features.sqlite3").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(core.InvalidEvidence):
        query_feature_store(root=tmp_path,limit=1)


def test_sqlite_builder_rejects_unknown_and_synthetic(tmp_path):
    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    row = {
        "site_id": site["site_id"],"model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],"source_type": "synthetic_demo",
        "feature_id": 1,"token_position":0,"activation":1.0,
        "sample_id":"p","prompt_group_id":"g",
    }
    with pytest.raises(ValueError,match="Synthetic"):
        create_feature_store([row],tmp_path/"one")
    with pytest.raises(ValueError,match="checkpoint"):
        create_feature_store([{**row,"source_type":"real_checkpoint","sae_sha256":""}],tmp_path/"two")



def test_published_dataset_rejects_modified_rows_and_fake_metadata(tmp_path, raw_rows):
    folder = tmp_path / "datasets" / "tinystories"
    pinned = "deadbeef0123456789deadbeef0123456789abcdef"
    freeze_dataset(
        "tinystories", raw_rows, source_revision=pinned, output=folder,
        license_reviewed=True, human_reviewed=True
    )
    rows = core.load_published_dataset("tinystories", tmp_path)
    assert rows is not None and len(rows) == 50
    assert sum(r["usage_split"] == "evaluation" for r in rows) == 30
    # An attacker/review mistake cannot silently change the two source populations.
    report = json.loads((folder / "manifest.json").read_text())
    report["source_repo"] = "impostor/dataset"
    (folder / "manifest.json").write_text(json.dumps(report))
    with pytest.raises(core.InvalidEvidence):
        core.load_published_dataset("tinystories", tmp_path)
    assert core.imported_counts(tmp_path) == {}


def test_source_group_sampling_ignores_input_record_iteration_order():
    # Duplicate prompts across distinct articles must be resolved by stable
    # rank, not whichever article was encountered first.
    raw = [
        {"id": "id" + str(i), "article_id": "a" + str(i), "text": "unique-" + str(i)}
        for i in range(80)
    ]
    raw[71]["text"] = raw[12]["text"]
    raw[73]["text"] = raw[18]["text"]
    one = select_records("wikitext103", raw)
    two = select_records("wikitext103", list(reversed(raw)))
    assert one == two
    assert len({x["prompt"] for x in one}) == 50


def test_exact_source_revision_is_required(raw_rows, tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        freeze_dataset(
            "tinystories", raw_rows, source_revision="main",
            output=tmp_path / "invalid", human_reviewed=True
        )
