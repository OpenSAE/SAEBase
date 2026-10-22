"""WP15 cards expose measured numbers and refuse unpublished text."""

from __future__ import annotations

import sys
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
