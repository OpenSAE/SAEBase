"""Explain stays a hypothesis unless a real generator is supplied."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tasks.explain.interpretation import propose_interpretation
from tasks.explain.operation import execute


def _rows(prefix: str, active: bool) -> dict:
    return {"group_id": prefix, "text": prefix + " text", "activation": 1.0 if active else 0.0}


def _cfg(tmp_path, **task):
    discovery = [_rows("d-on", True), _rows("d-off", False)]
    heldout = [_rows("h-on", True), _rows("h-off", False)]
    dfile = tmp_path / "discovery.json"
    hfile = tmp_path / "heldout.json"
    dfile.write_text(json.dumps(discovery), encoding="utf-8")
    hfile.write_text(json.dumps(heldout), encoding="utf-8")
    base = {
        "approved": True,
        "discovery_file": str(dfile),
        "heldout_file": str(hfile),
        "site_id": "blocks.25.hook_resid_post",
        "feature_id": 14325,
        "generator_model": "google/gemma-2-2b",
        "generator_method": "injected-test",
        "authorized_budget_usd": 0.0,
        "estimated_cost_usd": 0.0,
    }
    base.update(task)
    return SimpleNamespace(task=base)


def test_missing_generator_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="generator"):
        execute(_cfg(tmp_path))


def test_arbitrary_generator_name_is_not_executed(tmp_path):
    with pytest.raises(ValueError, match="allowlisted"):
        execute(_cfg(tmp_path, generator="os.system"))


def test_injected_generator_records_activation_agreement_not_semantics(tmp_path):
    result = execute(
        _cfg(tmp_path),
        generator=lambda snippets: "shared surface form",
        predictor=lambda description, text: text.startswith("h-on"),
    )
    hypothesis = result["hypothesis"]
    assert hypothesis["status"] == "heldout_tested"
    assert hypothesis["heldout_score"] == 1.0
    assert hypothesis["evaluation_kind"] == "SAE_binary_activation_prediction_not_semantic_truth"
    assert (
        "semantic" not in hypothesis["evaluation_kind"]
        or "not_semantic" in hypothesis["evaluation_kind"]
    )
    direct = propose_interpretation(
        site_id="blocks.25.hook_resid_post",
        feature_id=14325,
        discovery=[_rows("d-on", True)],
        heldout=[_rows("h-on", True), _rows("h-off", False)],
        generator=lambda snippets: "x",
        predictor=lambda description, text: text.startswith("h-on"),
        generator_model="test",
        generator_method="test",
        authorized_budget_usd=0,
        estimated_cost_usd=0,
    )
    assert direct["evaluation_kind"].endswith("not_semantic_truth")


def test_allowlisted_name_calls_saebase_generate(monkeypatch):
    from tasks.explain.local_generator import saebase_generate

    seen = {}

    def fake_generate(prompts, **kwargs):
        seen["prompts"] = prompts
        seen["kwargs"] = kwargs
        return [{"text": " They name places."}]

    monkeypatch.setattr("saebase.generate.generate", fake_generate)
    cfg = SimpleNamespace(
        task={
            "approved": True,
            "generator_model": "google/gemma-2-2b",
            "generator_max_new_tokens": 8,
            "generator_dtype": "bfloat16",
        }
    )
    text = saebase_generate(cfg)(["Paris is a city."])
    assert text == "They name places."
    assert seen["kwargs"]["model"] == "google/gemma-2-2b"
    assert seen["kwargs"]["dtype"] == "bfloat16"
    assert seen["kwargs"]["temperature"] == 0.0
    assert "Paris is a city." in seen["prompts"][0]


def test_allowlisted_generator_requires_model_and_approval():
    from tasks.explain.local_generator import saebase_generate

    with pytest.raises(PermissionError):
        saebase_generate(SimpleNamespace(task={"approved": False}))
    with pytest.raises(ValueError, match="generator_model"):
        saebase_generate(
            SimpleNamespace(
                task={
                    "approved": True,
                    "generator_model": "",
                    "generator_max_new_tokens": 8,
                }
            )
        )


def test_overlapping_heldout_groups_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        propose_interpretation(
            site_id="blocks.25.hook_resid_post",
            feature_id=1,
            discovery=[_rows("same", True)],
            heldout=[_rows("same", False)],
            generator=lambda snippets: "x",
            generator_model="test",
            generator_method="test",
            authorized_budget_usd=0,
            estimated_cost_usd=0,
        )
