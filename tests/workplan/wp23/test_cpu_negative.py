"""Target declarations are never promoted to fake native success."""

from tasks.precheck.compatibility_checks import declared_site, plan_compatibility


def test_missing_components_block_only_dependents():
    site = declared_site("model1", "sae1", 2, "blocks.2.hook_resid_post")
    states = {
        "dataset:data1": {"status": "BLOCKED"},
        "dataset:data2": {"status": "PASS"},
        "model:model1": {"status": "PASS"},
        "sae:model1:sae1:L2": {"status": "PASS"},
    }
    plan = plan_compatibility(["data1", "data2"], ["model1"], [site], states)
    assert len(plan["pairs"]) == 4
    assert sum(row["status"] == "BLOCKED" for row in plan["pairs"]) == 2
    assert sum(row["eligible"] for row in plan["pairs"]) == 2
    assert all(row["status"] != "PASS" for row in plan["pairs"])
