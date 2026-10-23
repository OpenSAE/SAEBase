"""No GPU, network, pretrained weights or synthetic-as-real results required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend import core  # noqa: E402
from backend.feature_store import create_feature_store, query_feature_store, feature_store_count  # noqa: E402
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
            assert output["reference_answers"] == ["4"]
        if source == "hellaswag":
            assert output["candidate_texts"] == row["endings"]
            assert "not the official" in output["evaluation_warning"]
        if source == "arc_challenge":
            assert output["candidate_texts"] == ["rock", "steam"]
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
        "tinystories",
        raw_rows,
        source_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        output=output,
    )
    assert manifest["sample_count"] == 50
    assert not manifest["human_reviewed"]
    assert core.imported_counts(tmp_path) == {}
    assert core.digest(output / "records.jsonl") == manifest["records_sha256"]
    with pytest.raises(FileExistsError):
        freeze_dataset(
            "tinystories",
            raw_rows,
            source_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            output=output,
        )
    reviewed = tmp_path / "datasets" / "boolq"
    inputs = [
        {"passage": "Unique passage " + str(i), "question": "Yes?", "answer": i % 2 == 0}
        for i in range(80)
    ]
    freeze_dataset(
        "boolq",
        inputs,
        source_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
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
        {**sample, "activation": (n + 1) / 10, "sample_id": "boolq:row" + str(n)} for n in range(17)
    ]
    rows.extend([{**sample, "feature_id": 7, "public_text_reviewed": False, "activation": 2}])
    result = create_feature_store(rows, tmp_path / "features")
    assert result["feature_count"] == 2
    assert result["observation_count"] == 18
    assert result["public_examples_stored"] == 8
    assert feature_store_count(tmp_path) == 2
    records = query_feature_store(root=tmp_path, query=":F6", limit=5)
    assert records["total"] == 1
    assert records["items"][0]["observations"] == 17
    assert records["items"][0]["unique_prompt_groups"] == 1
    assert len(records["items"][0]["examples"]) == 8
    # SQL wildcards do not alter exact literal searches.
    assert query_feature_store(root=tmp_path, query="%")["total"] == 0
    assert query_feature_store(root=tmp_path, query="_____%")["total"] == 0
    with pytest.raises(FileExistsError):
        create_feature_store([sample], tmp_path / "features")
    with (tmp_path / "features" / "features.sqlite3").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(core.InvalidEvidence):
        query_feature_store(root=tmp_path, limit=1)


def test_sqlite_builder_rejects_unknown_and_synthetic(tmp_path):
    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    row = {
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "source_type": "synthetic_demo",
        "feature_id": 1,
        "token_position": 0,
        "activation": 1.0,
        "sample_id": "p",
        "prompt_group_id": "g",
    }
    with pytest.raises(ValueError, match="Synthetic"):
        create_feature_store([row], tmp_path / "one")
    with pytest.raises(ValueError, match="checkpoint"):
        create_feature_store(
            [{**row, "source_type": "real_checkpoint", "sae_sha256": ""}], tmp_path / "two"
        )


def test_published_dataset_rejects_modified_rows_and_fake_metadata(tmp_path, raw_rows):
    folder = tmp_path / "datasets" / "tinystories"
    pinned = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    freeze_dataset(
        "tinystories",
        raw_rows,
        source_revision=pinned,
        output=folder,
        license_reviewed=True,
        human_reviewed=True,
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
            "tinystories",
            raw_rows,
            source_revision="main",
            output=tmp_path / "invalid",
            human_reviewed=True,
        )


def test_full_sequence_readout_and_unambiguous_task_scores():
    from backend.readouts import CandidateEvidence, score_continuations

    a = CandidateEvidence("A", (10, 11, 12), (-0.1, -0.1, -0.1))
    b = CandidateEvidence("B", (20,), (-0.2,))
    summed = score_continuations([a, b], aggregation="sum", gold_label="B")
    averaged = score_continuations([a, b], aggregation="mean", gold_label="B")
    assert summed["predicted_label"] == "B" and summed["correct"] is True
    assert averaged["predicted_label"] == "A" and averaged["correct"] is False
    assert all(row["n_tokens"] > 0 for row in summed["scores"])
    with pytest.raises(ValueError, match="nonpositive"):
        CandidateEvidence("bad", (3,), (0.5,))
    with pytest.raises(ValueError, match="complete"):
        CandidateEvidence("bad", (2, 3), (-0.5,))
    with pytest.raises(ValueError, match="unique"):
        score_continuations([a, a])


def test_input_anchor_prevents_invalid_rss_token_alignment():
    from backend.readouts import input_anchor

    base = "Question: Is the sky blue?"
    changed = "Background information.\nQuestion: Is the sky blue?"
    anchor = "Question: Is the sky blue?"
    start = len(changed) - len(anchor)
    mapping = [(0, 0), (0, 11), (11, start), (start, start + 9), (start + 9, len(changed))]
    # Variant tokenizer offsets describe precisely the unchanged question span.
    result = input_anchor(base, changed, anchor=anchor, offset_mapping=mapping)
    assert result["token_positions"] == [3, 4]
    assert result["variant_anchor_span"][0] == start
    with pytest.raises(ValueError, match="exactly once"):
        input_anchor(base, anchor + "\n" + anchor, anchor=anchor)
    with pytest.raises(ValueError, match="cuts through"):
        input_anchor(base, changed, anchor=anchor, offset_mapping=[(start - 1, len(changed))])


def test_cpu_sar_adapter_calls_only_pinned_trusted_estimator(tmp_path):
    from backend.methods import run_sar

    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    source = tmp_path / "sar.py"
    source.write_text(
        "def sar(g, masks, fit, held, g_full=None):\n"
        "    assert len(g) == 384 and len(fit) == 256 and len(held) == 128\n"
        "    return {'sar1': 0.5, 'rho2': float('nan')}\n"
    )
    trusted_sha = core.digest(source)
    input_row = {
        "source_type": "real_checkpoint",
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "prompt_group_id": "boolq:source1",
        "dataset_release": "curated-v1",
        "protocol_id": "sar.original.k16.m384.fit256.v1",
        "masks": [[False] * 16 for _ in range(384)],
        "g": [0.1 * i for i in range(384)],
        "fit": list(range(256)),
        "held": list(range(256, 384)),
        "g_full": 0.0,
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        run_sar(input_row, original=source, expected_sha="0" * 64)
    result = run_sar(input_row, original=source, expected_sha=trusted_sha)
    assert result["results"][0]["metric"] == "sar1"
    assert result["results"][0]["value"] == 0.5
    assert result["results"][1] == {"metric": "rho2", "value": None, "status": "undefined"}
    assert result["source_sha256"] == trusted_sha
    from backend.methods import build_analysis_index

    index = build_analysis_index([result], tmp_path / "analyses")
    assert index["count"] == 1
    assert index["reviewed_scientific_measurements"] is False
    assert core.analyses(tmp_path)[0]["kind"] == "sar"
    with pytest.raises(ValueError, match="Unverified|unverified"):
        build_analysis_index(
            [{**result, "measurement_input_sha256": "oops"}], tmp_path / "bad_analysis"
        )
    with pytest.raises(ValueError, match="literal booleans"):
        run_sar(
            {**input_row, "masks": [[1] * 16 for _ in range(384)]},
            original=source,
            expected_sha=trusted_sha,
        )
    # A mocked trusted function tests dispatch, not scientific parity with
    # the separate, original SAR paper implementation.
    with pytest.raises(ValueError, match="mismatch"):
        run_sar({**input_row, "sae_sha256": "0" * 64}, original=source, expected_sha=trusted_sha)


def test_cpu_rss_adapter_respects_held_out_variant_ids(tmp_path):
    from backend.methods import run_rss

    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    source = tmp_path / "rss.py"
    source.write_text(
        "import numpy as np\n"
        "def rss(delta):\n"
        "    return np.array([1]), np.array([1]), 0.25\n"
        "def held_out_rate(delta, support, direction):\n"
        "    return np.array([0.7])\n"
    )
    trusted_sha = core.digest(source)
    row = {
        "source_type": "real_checkpoint",
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "prompt_group_id": "boolq:source1",
        "dataset_release": "curated-v1",
        "protocol_id": "rss.transform.v1",
        "selection_variant_ids": [str(i) for i in range(5)],
        "held_variant_ids": [str(i) for i in range(5, 15)],
        "delta_selection": [[0.1, 0.2] for _ in range(5)],
        "delta_held": [[0.1, 0.2] for _ in range(10)],
    }
    record = run_rss(row, original=source, expected_sha=trusted_sha)
    assert record["support_ids"] == [1]
    assert record["results"][0]["value"] == 0.25
    assert record["results"][1]["value"] == 0.7
    with pytest.raises(ValueError, match="disjoint"):
        run_rss(
            {**row, "held_variant_ids": row["selection_variant_ids"] + ["x"] * 5},
            original=source,
            expected_sha=trusted_sha,
        )


def test_cpu_plan_has_exact_632_jobs_without_claiming_execution(tmp_path):
    from backend.plan import build_compute_plan

    plan = build_compute_plan(root=tmp_path)
    assert plan["schema"] == "opensae.compute_plan.v1"
    assert plan["site_count"] == 48
    assert plan["site_count_with_full_weight_hashes"] == 45
    assert plan["job_counts"] == {
        "capture": 480,
        "intervention": 60,
        "sar": 30,
        "rss": 60,
        "localization": 2,
    }
    assert len(plan["jobs"]) == 632
    assert plan["statuses"] == {"blocked": 632}
    assert plan["estimated_sar_coalition_measurements_if_all_frozen"] == 576000
    assert len({job["job_id"] for job in plan["jobs"]}) == 632
    assert all(job["execution"] == "not_run" for job in plan["jobs"])
    assert all("prompt" not in job["spec"] for job in plan["jobs"])


def test_cpu_plan_unblocks_only_pinned_reviewed_data_and_complete_sites(tmp_path, raw_rows):
    from backend.plan import build_compute_plan

    folder = tmp_path / "datasets" / "tinystories"
    freeze_dataset(
        "tinystories",
        raw_rows,
        source_revision="deadbeef" * 5,
        output=folder,
        license_reviewed=True,
        human_reviewed=True,
    )
    plan = build_compute_plan(root=tmp_path)
    assert plan["statuses"]["ready_for_external_gpu_operator"] == 45
    assert plan["statuses"]["blocked"] == 587
    assert (
        len(
            [
                j
                for j in plan["jobs"]
                if j["spec"]["task"] == "capture"
                and j["spec"]["dataset_id"] == "tinystories"
                and j["status"] == "blocked"
            ]
        )
        == 3
    )
    assert (
        len(
            [
                j
                for j in plan["jobs"]
                if j["spec"]["task"] == "sar" and j["status"] == "ready_for_external_gpu_operator"
            ]
        )
        == 0
    )


def test_cpu_source_acquisition_is_pinned_and_never_public_without_review(tmp_path, monkeypatch):
    import types
    from backend.acquire import acquire

    original_rows = [{"id": str(i), "text": "A unique short story " + str(i)} for i in range(120)]

    class FakeStream:
        def shuffle(self, *, seed, buffer_size):
            assert seed == 20261018
            assert buffer_size == 2048
            return self

        def __iter__(self):
            return iter(original_rows)

    def fake_load_dataset(**kwargs):
        assert kwargs["path"] == "roneneldan/TinyStories"
        assert kwargs["split"] == "validation"
        assert kwargs["revision"] == "f" * 40
        assert "trust_remote_code" not in kwargs
        return FakeStream()

    class FakeHfApi:
        def dataset_info(self, *, repo_id):
            assert repo_id == "roneneldan/TinyStories"
            return types.SimpleNamespace(sha="f" * 40)

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    path = tmp_path / "source.jsonl"
    result = acquire("tinystories", path, max_raw_rows=100)
    assert result["resolved_revision"] == "f" * 40
    assert result["license_reviewed"] is False
    assert result["auto_publish"] is False
    assert core.digest(path) == result["source_export_sha256"]
    assert len(path.read_text().strip().splitlines()) == 100
    with pytest.raises(FileExistsError):
        acquire("tinystories", path, max_raw_rows=100)


def test_rss_cpu_transformations_keep_parent_and_disjoint_roles():
    from backend.transforms import build_variants

    kwargs = {
        "dataset_id": "boolq",
        "sample_id": "boolq:s1",
        "prompt_group_id": "boolq:g1",
        "base_prompt": "Passage:\nCats sleep.\nQuestion: Do cats sleep?\nAnswer yes or no:",
        "anchor": "Question: Do cats sleep?",
        "source_revision": "d" * 40,
    }
    result = build_variants(**kwargs)
    assert result["n_proposals"] == 45
    assert result["fully_reviewed"] is False
    assert result == build_variants(**kwargs)
    variants = result["variants"]
    assert len({x["variant_id"] for x in variants}) == 45
    for setting in ("local_format", "instruction_wrapper", "context_wrapper"):
        rows = [x for x in variants if x["setting"] == setting]
        assert len(rows) == 15
        assert sum(x["variant_role"] == "selection" for x in rows) == 5
        assert sum(x["variant_role"] == "held_out" for x in rows) == 10
        assert len({x["operator"] for x in rows}) == 5
        assert all(x["prompt_group_id"] == "boolq:g1" for x in rows)
        assert all(x["semantic_review"] == "required" for x in rows)
        assert all(x["text"].count(kwargs["anchor"]) == 1 for x in rows)
    changed = build_variants(**{**kwargs, "source_revision": "a" * 40})
    assert result["variants"][0]["variant_id"] != changed["variants"][0]["variant_id"]


def test_rss_variant_builder_supports_all_ten_sources_as_unreviewed_proposals():
    from backend.transforms import build_variants

    for source in [row["id"] for row in core.dataset_catalog()["sources"]]:
        base = (
            'def example(x):\n    """Describe x."""\n'
            if source == "humaneval"
            else "A standalone original question about a measurable model feature."
        )
        variants = build_variants(
            source,
            source + ":id",
            source + ":group",
            base,
            anchor=base,
            source_revision="a" * 40,
        )
        assert variants["n_proposals"] == 45
        assert variants["fully_reviewed"] is False
        assert len({v["variant_id"] for v in variants["variants"]}) == 45
        assert all(v["semantic_review"] == "required" for v in variants["variants"])
        assert all(v["tokenizer_alignment"] == "pending" for v in variants["variants"])
        assert all(v["text"].count(base) == 1 for v in variants["variants"])
        if source == "humaneval":
            assert variants["transform_family"] == "python_comment_and_whitespace"
            assert all(v["text"].lstrip().startswith(("#", "def")) for v in variants["variants"])
        if source in ("wikitext103", "tinystories"):
            assert variants["transform_family"] == "prose_continuation"
    with pytest.raises(ValueError, match="exactly|once|Anchor"):
        build_variants(
            "boolq",
            "boolq:id",
            "boolq:group",
            "Answer: Answer:",
            anchor="Answer:",
            source_revision="a" * 40,
        )


def test_durable_queue_idempotency_private_reference_and_lease_recovery(tmp_path):
    from backend.jobs import JobError, JobQueue

    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    request = {
        "site_id": site["site_id"],
        "operation": "ablate",
        "feature_ids": [1, 2],
        "strength": 1.0,
        "max_new_tokens": 16,
        "input_ref": "a" * 64,
    }
    q = JobQueue(tmp_path / "private" / "queue.sqlite3", max_queued=1, max_per_owner=1)
    first = q.enqueue(owner="research-user-1", client_key="nonce-1234", spec=request, now=100)
    assert first["status"] == "queued" and not first["deduplicated"]
    same = q.enqueue(owner="research-user-1", client_key="nonce-1234", spec=request, now=101)
    assert same["job_id"] == first["job_id"] and same["deduplicated"]
    assert q.status(job_id=first["job_id"], owner="unrelated-user") is None
    with pytest.raises(JobError, match="quota"):
        q.enqueue(owner="research-user-2", client_key="nonce-5678", spec=request, now=102)
    with pytest.raises(JobError, match="different inputs"):
        q.enqueue(
            owner="research-user-1",
            client_key="nonce-1234",
            spec={**request, "strength": 2.0},
            now=103,
        )
    with pytest.raises(JobError, match="raw prompt"):
        q.enqueue(
            owner="research-user-1",
            client_key="nonce-9999",
            spec={**request, "text": "secret original prompt"},
            now=104,
        )
    claim = q.claim(worker_id="gpu-worker-1", now=110, lease_seconds=30)
    assert claim["job_id"] == first["job_id"] and claim["attempt"] == 1
    assert (
        q.heartbeat(
            job_id=claim["job_id"], worker_id="wrong-worker", attempt=claim["attempt"], now=111
        )
        is False
    )
    assert q.heartbeat(
        job_id=claim["job_id"],
        worker_id="gpu-worker-1",
        attempt=claim["attempt"],
        now=120,
        lease_seconds=30,
    )
    assert q.reap_expired(now=151) == {"requeued": 1, "failed": 0}
    claim2 = q.claim(worker_id="gpu-worker-2", now=152, lease_seconds=30)
    assert claim2["attempt"] == 2
    assert not q.finish(
        job_id=claim2["job_id"],
        worker_id="gpu-worker-1",
        attempt=claim["attempt"],
        result_sha256="f" * 64,
        now=155,
    )
    assert q.finish(
        job_id=claim2["job_id"],
        worker_id="gpu-worker-2",
        attempt=claim2["attempt"],
        result_sha256="f" * 64,
        now=155,
    )
    assert q.status(job_id=first["job_id"], owner="research-user-1")["status"] == "succeeded"
    assert q.cancel(job_id=first["job_id"], owner="research-user-1") is False
    assert "secret original prompt" not in (
        tmp_path / "private" / "queue.sqlite3"
    ).read_bytes().decode("utf-8", errors="ignore")


def test_queue_refuses_invalid_specs_and_stale_worker_completion(tmp_path):
    from backend.jobs import JobError, JobQueue

    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    q = JobQueue(tmp_path / "jobs.sqlite3")
    spec = {
        "site_id": site["site_id"],
        "operation": "identity",
        "feature_ids": [],
        "strength": 1,
        "max_new_tokens": 16,
        "input_ref": "a" * 64,
    }
    with pytest.raises(JobError, match="64-character|private content hash"):
        q.enqueue(owner="owner-1", client_key="nonce-123456", spec={**spec, "input_ref": "bad"})
    with pytest.raises(JobError, match="Identity|feature IDs"):
        q.enqueue(owner="owner-1", client_key="nonce-123456", spec={**spec, "feature_ids": [1]})
    job = q.enqueue(owner="owner-1", client_key="nonce-123456", spec=spec, now=100)
    assert q.claim(worker_id="worker-1", now=101, lease_seconds=30)["attempt"] == 1
    assert (
        q.finish(
            job_id=job["job_id"], worker_id="worker-1", attempt=1, error_code="MODEL_ERROR", now=132
        )
        is False
    )
    assert q.reap_expired(now=132, max_attempts=1)["failed"] == 1
    assert q.status(job_id=job["job_id"], owner="owner-1")["status"] == "failed"


def test_controlled_six_factor_designs_are_complete_and_original_compatible(tmp_path):
    from backend.controlled_designs import (
        DESIGNS,
        check_candidate,
        make_candidate,
        to_saebase_design,
        write_candidate,
    )

    configuration = json.loads((PROJECT / "data" / "coverage.json").read_text(encoding="utf-8"))
    assert set(configuration["localize_designs"]) == set(DESIGNS)
    assert len(DESIGNS) == 2
    for name in DESIGNS:
        candidate = make_candidate(name)
        assert candidate == make_candidate(name)
        assert check_candidate(candidate) == candidate
        assert candidate["status"] == "input_only_no_model_execution"
        assert candidate["n_factors"] == 6
        assert candidate["n_states"] == 64
        assert candidate["independent_group_count"] == 1
        assert [row["mask"] for row in candidate["states"]] == list(range(64))
        assert len({row["prompt"] for row in candidate["states"]}) == 64
        assert len({row["sample_id"] for row in candidate["states"]}) == 64
        assert all(
            row["bits"] == [(row["mask"] >> bit) & 1 for bit in range(6)]
            for row in candidate["states"]
        )
        assert all(row["token_position"] == -1 for row in candidate["states"])
        actual = to_saebase_design(candidate)
        assert actual.n_factors == 6
        assert actual.n_states == 64
        assert actual.sample_ids_by_state == {
            row["mask"]: row["sample_id"] for row in candidate["states"]
        }
        assert actual.prompts_by_state == {
            row["mask"]: row["prompt"] for row in candidate["states"]
        }
        output = tmp_path / (name + ".json")
        result = write_candidate(name, output)
        assert result["model_inferences"] == 0
        assert result["n_states"] == 64
        assert check_candidate(json.loads(output.read_text(encoding="utf-8"))) == candidate
        with pytest.raises(FileExistsError):
            write_candidate(name, output)


def test_synthetic_control_is_not_presented_as_real_sae_data():
    from backend.controlled_designs import (
        check_candidate,
        make_candidate,
        synthetic_reference_codes,
    )

    controlled = make_candidate("synthetic_known_factor_control")
    synthetic = synthetic_reference_codes(controlled)
    assert synthetic["source_type"] == "synthetic_logic_not_real_sae"
    assert synthetic["n_states"] == 64
    assert synthetic["n_synthetic_features"] == 6
    assert len(synthetic["feature_values"]) == 64
    assert all(len(values) == 6 for values in synthetic["feature_values"])
    assert synthetic["feature_values"][0] == [0.0] * 6
    assert synthetic["feature_values"][63] == [1.0, 2.0, 1.0, 1.0, 0.0, 2.0]
    assert synthetic_reference_codes(controlled) == synthetic
    tampered = json.loads(json.dumps(controlled))
    tampered["states"][16]["prompt"] = "Not the original controlled input."
    with pytest.raises(ValueError, match="mismatch"):
        check_candidate(tampered)
    with pytest.raises(ValueError, match="Synthetic truth"):
        synthetic_reference_codes(make_candidate("real_model_six_factor_design"))


def test_wikitext_raw_lines_preserve_original_article_membership(monkeypatch, tmp_path):
    """Raw-line shuffling before grouping would attach passages to wrong articles."""
    import types
    from backend.acquire import acquire

    lines = [
        {"text": "= Alpha ="},
        {"text": "A" * 95},
        {"text": "= Beta ="},
        {"text": "B" * 95},
        {"text": "= Gamma ="},
        {"text": "C" * 95},
    ]

    class Stream:
        def __iter__(self):
            return iter(lines)

        def shuffle(self, **kwargs):
            raise AssertionError("Line-level WikiText must not be shuffled")

    class FakeApi:
        def dataset_info(self, *, repo_id, revision=None):
            assert repo_id == "Salesforce/wikitext"
            if revision is not None:
                assert revision == "f" * 40
            return types.SimpleNamespace(sha="f" * 40)

    def load_dataset(**kwargs):
        assert kwargs["revision"] == "f" * 40
        assert kwargs["name"] == "wikitext-103-raw-v1"
        return Stream()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    target = tmp_path / "wikitext.jsonl"
    result = acquire(
        "wikitext103",
        target,
        max_raw_rows=100,
        source_revision="f" * 40,
    )
    entries = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 3
    assert entries[0]["article_id"] != entries[1]["article_id"]
    assert entries[1]["article_id"] != entries[2]["article_id"]
    assert result["shuffle_buffer_size"] == 0
    assert result["resolved_revision"] == "f" * 40


def test_redacted_cpu_audit_never_releases_unreviewed_prompt_text(tmp_path, monkeypatch):
    import hashlib
    from backend import source_audit

    raw = [{"id": str(i), "text": "Confidential candidate story text " + str(i)} for i in range(65)]

    def fake_acquire(dataset_id, destination, **kwargs):
        assert dataset_id == "tinystories"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
        return {
            "repo": "roneneldan/TinyStories",
            "config": None,
            "split": "validation",
            "resolved_revision": "b" * 40,
            "source_export_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "sampling_scope": "test-only source export",
            "max_raw_rows": 2500,
        }

    monkeypatch.setattr(source_audit, "acquire", fake_acquire)
    report = source_audit.audit_sources(
        destination=tmp_path / "private_manifest.json",
        dataset_ids=["tinystories"],
    )
    assert report["sources_selected"] == 1
    assert report["candidate_count"] == 50
    assert report["gpu_calls"] == 0
    assert report["rights_review"] == "not_completed"
    assert report["full_source_text_in_this_artifact"] is False
    data = (tmp_path / "private_manifest.json").read_text(encoding="utf-8")
    assert "Confidential candidate story text" not in data
    assert all(
        len(row["text_sha256"]) == 64
        for row in report["dataset_results"][0]["candidate_rows_without_text"]
    )
    assert all(
        "prompt" not in row for row in report["dataset_results"][0]["candidate_rows_without_text"]
    )


def test_corpus_source_lock_has_ten_exact_pinned_upstream_commits():
    from backend.corpus_lock import LOCK_PATH, SCHEMA

    source = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert source["schema"] == SCHEMA
    assert source["total_sources"] == 10
    assert source["total_base_groups"] == 500
    assert source["human_content_review"] == "pending"
    assert source["rights_review"] == "pending"
    assert source["candidate_rows_in_repository"] is False
    assert len({row["dataset_id"] for row in source["sources"]}) == 10
    assert all(
        len(row["source_revision"]) == 40
        and len(row["candidate_sha256"]) == 64
        and len(row["source_export_sha256"]) == 64
        and row["sample_count"] == 50
        for row in source["sources"]
    )


def test_redacted_corpus_lock_verification_rejects_changed_content():
    import hashlib
    from backend.corpus_lock import verify_corpus_lock

    dataset_ids = [row["id"] for row in core.dataset_catalog()["sources"]]
    rows = []
    frozen = []
    for dataset_id in dataset_ids:
        revision = "b" * 40
        source_sha = hashlib.sha256((dataset_id + "source").encode()).hexdigest()
        candidate_sha = hashlib.sha256((dataset_id + "candidate").encode()).hexdigest()
        group_rows = [
            {
                "dataset_id": dataset_id,
                "source_revision": revision,
                "sample_id": dataset_id + ":sample:" + str(i),
                "prompt_group_id": dataset_id + ":group:" + str(i),
                "usage_split": "discovery" if i < 20 else "evaluation",
                "text_sha256": hashlib.sha256((dataset_id + str(i)).encode()).hexdigest(),
            }
            for i in range(50)
        ]
        frozen.append(
            {
                "dataset_id": dataset_id,
                "status": "candidate_frozen_unreviewed",
                "source_revision": revision,
                "source_export_sha256": source_sha,
                "candidate_sha256": candidate_sha,
                "max_raw_rows": 2500,
                "n_candidates": 50,
                "discovery": 20,
                "evaluation": 30,
                "license_reviewed": False,
                "human_reviewed": False,
                "candidate_rows_without_text": group_rows,
            }
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "source_revision": revision,
                "source_export_sha256": source_sha,
                "candidate_sha256": candidate_sha,
                "max_raw_rows": 2500,
            }
        )
    report = {
        "schema": "opensae.corpus_cpu_audit.v1",
        "sources_selected": 10,
        "candidate_count": 500,
        "number_sources_target": 10,
        "gpu_calls": 0,
        "full_source_text_in_this_artifact": False,
        "cross_source_exact_duplicates": [],
        "dataset_results": frozen,
    }
    report["manifest_sha256"] = hashlib.sha256(core.canonical_json(report)).hexdigest()
    lock = {
        "schema": "opensae.corpus_source_lock.v1",
        "total_sources": 10,
        "total_base_groups": 500,
        "discovery_groups": 200,
        "evaluation_groups": 300,
        "rights_review": "pending",
        "human_content_review": "pending",
        "audit_manifest_sha256": report["manifest_sha256"],
        "sources": rows,
    }
    result = verify_corpus_lock(report, lock)
    assert result["status"] == "frozen_candidates_verified_unreviewed"
    assert result["groups"] == 500
    tampered = json.loads(json.dumps(report))
    tampered["dataset_results"][0]["candidate_rows_without_text"][0]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        verify_corpus_lock(tampered, lock)
    tampered["dataset_results"][0]["candidate_rows_without_text"][0]["prompt"] = "private original"
    with pytest.raises(ValueError, match="prompt text"):
        verify_corpus_lock(tampered, lock, verify_aggregate=False)


def test_hellaswag_gold_labels_are_balanced_before_model_evaluation():
    candidates = [
        {
            "ctx": "An independent example with source number " + str(i),
            "endings": ["First ending.", "Second ending.", "Third ending.", "Fourth ending."],
            "label": str(i % 4),
        }
        for i in range(160)
    ]
    rows = select_records("hellaswag", candidates)
    assert len(rows) == 50
    from collections import Counter

    assert Counter(item["task_label"] for item in rows) == Counter(
        {"A": 13, "B": 13, "C": 12, "D": 12}
    )
    assert rows == select_records("hellaswag", list(reversed(candidates)))


def test_cpu_near_duplicate_screen_reports_only_sample_ids(tmp_path):
    from backend.near_duplicates import scan_near_duplicates

    base = "A careful research question asks the same central question about model features"
    source1 = tmp_path / "boolq"
    source2 = tmp_path / "squad"
    source1.mkdir()
    source2.mkdir()
    example_a = {
        "sample_id": "boolq:s1",
        "usage_split": "discovery",
        "prompt": base + " in its original context.",
    }
    example_b = {
        "sample_id": "squad:s2",
        "usage_split": "evaluation",
        "prompt": base + " in its original context!",
    }
    (source1 / "records.jsonl").write_text(json.dumps(example_a) + "\n", encoding="utf-8")
    (source2 / "records.jsonl").write_text(json.dumps(example_b) + "\n", encoding="utf-8")
    report = scan_near_duplicates(tmp_path, ["boolq", "squad"])
    assert report["lexical_near_duplicate_count"] == 1
    assert report["candidates"][0]["cross_dataset"] is True
    assert report["candidates"][0]["cross_usage_split"] is True
    assert base not in json.dumps(report)
    assert report["interpretation"].endswith("human review pending")


def test_cpu_rss_proposal_audit_does_not_publish_third_party_text(tmp_path):
    from backend.variant_audit import summarize_variant_proposals

    root = tmp_path / "frozen" / "boolq"
    root.mkdir(parents=True)
    rows = [
        {
            "dataset_id": "boolq",
            "sample_id": "boolq:sample" + str(i),
            "prompt_group_id": "boolq:group" + str(i),
            "source_revision": "a" * 40,
            "prompt": "A unique original question asks whether a feature is active " + str(i),
        }
        for i in range(2)
    ]
    (root / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rows), encoding="utf-8"
    )
    summary = summarize_variant_proposals(tmp_path / "frozen", ["boolq"])
    assert summary == summarize_variant_proposals(tmp_path / "frozen", ["boolq"])
    assert summary["parent_groups"] == 2
    assert summary["proposed_variants"] == 90
    assert summary["selection_proposals"] == 30
    assert summary["held_out_proposals"] == 60
    assert summary["semantic_review"] == "pending"
    assert summary["input_texts_in_audit"] is False
    assert len(summary["metadata_only_sha256"]) == 64
    assert rows[0]["prompt"] not in json.dumps(summary)


def test_worker_attempt_fencing_rejects_stale_identical_worker_identity(tmp_path):
    from backend.jobs import JobQueue

    site = next(s for s in core.load_sites() if core.has_complete_identity(s))
    queue = JobQueue(tmp_path / "fenced.sqlite3")
    request = {
        "site_id": site["site_id"],
        "operation": "identity",
        "feature_ids": [],
        "strength": 1.0,
        "max_new_tokens": 16,
        "input_ref": "e" * 64,
    }
    job = queue.enqueue(owner="owner-retry", client_key="retry-1234", spec=request, now=100)
    first = queue.claim(worker_id="same-gpu-worker", now=101, lease_seconds=30)
    assert first["attempt"] == 1
    assert queue.reap_expired(now=132)["requeued"] == 1
    second = queue.claim(worker_id="same-gpu-worker", now=133, lease_seconds=30)
    assert second["attempt"] == 2
    assert (
        queue.heartbeat(
            job_id=job["job_id"],
            worker_id="same-gpu-worker",
            attempt=first["attempt"],
            now=134,
        )
        is False
    )
    assert (
        queue.finish(
            job_id=job["job_id"],
            worker_id="same-gpu-worker",
            attempt=first["attempt"],
            result_sha256="b" * 64,
            now=135,
        )
        is False
    )
    assert (
        queue.finish(
            job_id=job["job_id"],
            worker_id="same-gpu-worker",
            attempt=second["attempt"],
            result_sha256="c" * 64,
            now=135,
        )
        is True
    )
    assert queue.status(job_id=job["job_id"], owner="owner-retry")["status"] == "succeeded"


def test_registered_scope_assets_endpoint_is_metadata_only():
    response = client.get("/api/sae-assets?model_key=gemma270m")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "registered_metadata_only"
    assert data["total"] >= 3
    assert all(item["model_key"] == "gemma270m" for item in data["items"])
    assert all(item["status"] == "registered_not_live_verified" for item in data["items"])
    assert all(item["repo_id"].startswith("google/gemma-scope-2-") for item in data["items"])


def test_opensae_workspace_cors_is_allowlisted():
    """Browser API access is enabled only for the explicitly configured site."""
    allowed = "https://atheavenma.github.io"
    response = client.options(
        "/api/sites?limit=5",
        headers={
            "Origin": allowed,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == allowed
    response = client.options(
        "/api/private/jobs",
        headers={
            "Origin": allowed,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert response.status_code == 200
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
    assert response.headers["access-control-allow-private-network"] == "true"
    rejected = client.options(
        "/api/sites",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in rejected.headers
