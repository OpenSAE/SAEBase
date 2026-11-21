"""Offline negative cases for independently declared compatibility targets."""

import pytest

from tasks.precheck.compatibility_checks import declared_site, plan_compatibility


def test_plan_only_no_fabricated_pair_success():
    site = declared_site("model1", "sae1", 3, "blocks.3.hook_resid_post")
    empty = plan_compatibility(["ds1", "ds2"], ["model1"], [site])
    assert empty["model_dataset_targets"] == 2
    assert empty["sae_site_dataset_targets"] == 2
    assert len(empty["pairs"]) == 4
    assert all(row["status"] == "NOT_RUN" and not row["eligible"] for row in empty["pairs"])
    passed = {
        "dataset:ds1": {"status": "PASS"},
        "model:model1": {"status": "PASS"},
        "sae:model1:sae1:L3": {"status": "PASS"},
    }
    selected = plan_compatibility(["ds1", "ds2"], ["model1"], [site], passed)
    assert sum(row["eligible"] for row in selected["pairs"]) == 2
    assert all(row["status"] != "PASS" for row in selected["pairs"])
    passed["dataset:ds1"]["status"] = "BLOCKED"
    blocked = plan_compatibility(["ds1", "ds2"], ["model1"], [site], passed)
    assert sum(row["status"] == "BLOCKED" for row in blocked["pairs"]) == 2


def test_reject_duplicate_invalid_or_fake_site():
    site = declared_site("model1", "sae1", 3, "blocks.3.hook_resid_post")
    with pytest.raises(ValueError, match="Duplicate"):
        plan_compatibility(["ds1", "ds1"], ["model1"], [site])
    with pytest.raises(ValueError, match="Duplicate"):
        plan_compatibility(["ds1"], ["model1"], [site, site])
    with pytest.raises(ValueError, match="identity"):
        plan_compatibility(["ds1"], ["model1"], [{**site, "site_id": "fake"}])
    with pytest.raises(ValueError, match="layer"):
        declared_site("model1", "sae1", 3, "blocks.4.hook_resid_post")
    with pytest.raises(ValueError, match="unselected"):
        plan_compatibility(["ds1"], ["model2"], [site])


def test_empty_and_unsafe_scopes_rejected():
    with pytest.raises(ValueError):
        plan_compatibility([], ["model1"], [])
    with pytest.raises(ValueError):
        plan_compatibility(["../private"], ["model1"], [])
