"""WP15 cards expose measured numbers and refuse unpublished text."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import pytest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend.api import app  # noqa: E402
from backend.wp15_records import load_record  # noqa: E402

client = TestClient(app)


def test_recorded_cards_match_the_real_runs_and_hide_text():
    benchmark = load_record("benchmark")
    training = load_record("training")
    generation = load_record("generation")
    explanation = load_record("explanation")
    assert benchmark["metrics"]["saebench.core.l0"] == 376.0
    assert benchmark["metrics"]["saebench.core.mse"] == 0.02498825453221798
    assert benchmark["comparable"] is False
    assert training["checkpoint_sha256"].startswith("1c8169fa")
    assert training["reconstruction_mse"] == 7.346281051635742
    assert training["global_step"] == 50
    changed = generation["samples"][1]["conditions"][1]
    assert changed["same_continuation"] is False
    assert changed["argmax_changed"] is True
    assert changed["step0_max_abs_logit_delta"] == 9.802836418151855
    assert explanation["approved"] is False
    assert explanation["heldout_score"] == 0.5
    for name in ("benchmark", "training", "generation", "explanation"):
        response = client.get("/api/wp15/" + name)
        assert response.status_code == 200
        assert response.json()["rights_reviewed"] is False


def test_unknown_wp15_card_is_not_invented():
    assert client.get("/api/wp15/weights").status_code == 404


def test_reviewed_staging_bundle_requires_human_approval(tmp_path, monkeypatch):
    """A reviewed description can be staged without making unreviewed text public."""
    from backend import wp15_records

    source = wp15_records._root()
    staged = tmp_path / "wp15"
    shutil.copytree(source, staged)
    monkeypatch.setenv("OPENSAE_LAB_DATA", str(tmp_path))

    record_path = staged / "explanation.json"
    candidate = json.loads(record_path.read_text())
    candidate.update(
        {
            "approved": True,
            "human_reviewed": True,
            "rights_reviewed": True,
            "private_unpublished": False,
            "description_published": True,
            "description": "Test-only approved explanation for a simulated review flow.",
        }
    )
    record_path.write_text(json.dumps(candidate))
    returned = wp15_records.load_record("explanation")
    assert returned["description_published"] is True
    assert returned["description"] == candidate["description"]

    candidate["human_reviewed"] = False
    record_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        wp15_records.load_record("explanation")

    candidate["human_reviewed"] = True
    candidate["prompt_text"] = "Private material must never appear."
    record_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        wp15_records.load_record("explanation")


def test_staged_bundle_overrides_preview_only_when_present(tmp_path, monkeypatch):
    from backend import wp15_records

    source = wp15_records._root()
    staged = tmp_path / "wp15"
    shutil.copytree(source, staged)
    monkeypatch.setenv("OPENSAE_LAB_DATA", str(tmp_path))
    benchmark = json.loads((staged / "benchmark.json").read_text())
    benchmark["metrics"]["saebench.core.l0"] = 375.0
    (staged / "benchmark.json").write_text(json.dumps(benchmark))
    assert wp15_records.load_record("benchmark")["metrics"]["saebench.core.l0"] == 375.0


def test_generation_can_publish_only_source_matched_reviewed_examples(tmp_path, monkeypatch):
    from backend import wp15_records

    source = wp15_records._root()
    staged = tmp_path / "wp15"
    shutil.copytree(source, staged)
    monkeypatch.setenv("OPENSAE_LAB_DATA", str(tmp_path))
    path = staged / "generation.json"
    record = json.loads(path.read_text())
    record["rights_reviewed"] = True
    record["public_prompt"] = True
    record["public_continuations_reviewed"] = True
    for idx, sample in enumerate(record["samples"]):
        prompt = "Original demo prompt " + str(idx)
        sample["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        sample["public_example"] = {
            "prompt_text": prompt,
            "baseline_continuation": " Baseline",
            "edited_continuation": " Edited",
        }
    path.write_text(json.dumps(record))
    accepted = wp15_records.load_record("generation")
    assert accepted["samples"][0]["public_example"]["prompt_text"].startswith("Original")

    record["samples"][0]["public_example"]["prompt_text"] = "Source text modified"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="match original GPU source"):
        wp15_records.load_record("generation")

    record["public_prompt"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        wp15_records.load_record("generation")
