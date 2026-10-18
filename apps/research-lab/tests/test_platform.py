"""No GPU, network, pretrained weights or synthetic-as-real results required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend import core  # noqa: E402
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
        "tinystories", raw_rows, source_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", output=output
    )
    assert manifest["sample_count"] == 50
    assert not manifest["human_reviewed"]
    assert core.imported_counts(tmp_path) == {}
    assert core.digest(output / "records.jsonl") == manifest["records_sha256"]
    with pytest.raises(FileExistsError):
        freeze_dataset("tinystories", raw_rows, source_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", output=output)
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
    assert query_feature_store(root=tmp_path,query="_____%")["total"] == 0
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
    pinned = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
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
        "source_type": "real_checkpoint", "site_id": site["site_id"],
        "model_sha256": site["model_sha256"], "sae_sha256": site["sae_sha256"],
        "prompt_group_id": "boolq:source1", "dataset_release": "curated-v1",
        "protocol_id": "sar.original.k16.m384.fit256.v1",
        "masks": [[False] * 16 for _ in range(384)],
        "g": [0.1 * i for i in range(384)],
        "fit": list(range(256)), "held": list(range(256, 384)), "g_full": 0.0,
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
        build_analysis_index([{**result, "measurement_input_sha256": "oops"}], tmp_path / "bad_analysis")
    with pytest.raises(ValueError, match="literal booleans"):
        run_sar({**input_row, "masks": [[1] * 16 for _ in range(384)]}, original=source, expected_sha=trusted_sha)
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
        "source_type": "real_checkpoint", "site_id": site["site_id"],
        "model_sha256": site["model_sha256"], "sae_sha256": site["sae_sha256"],
        "prompt_group_id": "boolq:source1", "dataset_release": "curated-v1",
        "protocol_id": "rss.transform.v1",
        "selection_variant_ids": [str(i) for i in range(5)],
        "held_variant_ids": [str(i) for i in range(5,15)],
        "delta_selection": [[0.1, 0.2] for _ in range(5)],
        "delta_held": [[0.1, 0.2] for _ in range(10)],
    }
    record = run_rss(row, original=source, expected_sha=trusted_sha)
    assert record["support_ids"] == [1]
    assert record["results"][0]["value"] == 0.25
    assert record["results"][1]["value"] == 0.7
    with pytest.raises(ValueError, match="disjoint"):
        run_rss({**row, "held_variant_ids": row["selection_variant_ids"] + ["x"] * 5}, original=source, expected_sha=trusted_sha)



def test_cpu_plan_has_exact_632_jobs_without_claiming_execution(tmp_path):
    from backend.plan import build_compute_plan

    plan = build_compute_plan(root=tmp_path)
    assert plan["schema"] == "opensae.compute_plan.v1"
    assert plan["site_count"] == 48
    assert plan["site_count_with_full_weight_hashes"] == 45
    assert plan["job_counts"] == {
        "capture": 480, "intervention": 60, "sar": 30,
        "rss": 60, "localization": 2
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
        "tinystories", raw_rows,
        source_revision="deadbeef" * 5, output=folder,
        license_reviewed=True, human_reviewed=True
    )
    plan = build_compute_plan(root=tmp_path)
    assert plan["statuses"]["ready_for_external_gpu_operator"] == 45
    assert plan["statuses"]["blocked"] == 587
    assert len([j for j in plan["jobs"] if j["spec"]["task"] == "capture"
                and j["spec"]["dataset_id"] == "tinystories"
                and j["status"] == "blocked"]) == 3
    assert len([j for j in plan["jobs"] if j["spec"]["task"] == "sar"
                and j["status"] == "ready_for_external_gpu_operator"]) == 0



def test_cpu_source_acquisition_is_pinned_and_never_public_without_review(tmp_path, monkeypatch):
    import types
    from backend.acquire import acquire

    original_rows = [
        {"id": str(i), "text": "A unique short story " + str(i)}
        for i in range(120)
    ]

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

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
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


def test_rss_variant_builder_blocks_code_and_continuation_templates():
    from backend.transforms import build_variants

    for dataset in ("humaneval", "wikitext103", "tinystories"):
        with pytest.raises(ValueError, match="independently designed"):
            build_variants(
                dataset, dataset + ":id", dataset + ":group", "A sample text.",
                anchor="sample", source_revision="a" * 40,
            )
    with pytest.raises(ValueError, match="exactly|once|Anchor"):
        build_variants(
            "boolq", "boolq:id", "boolq:group", "Answer: Answer:",
            anchor="Answer:", source_revision="a" * 40,
        )
