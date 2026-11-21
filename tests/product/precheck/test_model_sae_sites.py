"""No mocked forward may claim a native Model PASS."""

from tasks.precheck.model_checks import check_real_model


def test_empty_prompt_does_not_load_or_pass():
    result = check_real_model(
        model_key="gemma270m",
        model_name="gemma-3-270m",
        prompt="",
        device="cpu",
    )
    assert result["status"] == "BLOCKED"
    assert result["code"] == "BLOCKED_CONFIGURATION"
    assert (
        result["scientific_acceptance"] == "NOT_COVERED"
        if "scientific_acceptance" in result
        else True
    )
