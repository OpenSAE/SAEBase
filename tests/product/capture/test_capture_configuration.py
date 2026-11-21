"""Synthetic capture regression: configuration changes output before any weight loading."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tasks.capture import operation


@pytest.mark.parametrize("requested, expected", [(1, [1]), (3, [1, 3, 2]), (20, [1, 3, 2, 0])])
def test_top_feature_num_controls_actual_capture_output(monkeypatch, requested, expected):
    import saebase.api

    pair = SimpleNamespace(
        model=object(),
        sae=object(),
        site=SimpleNamespace(hook_name="synthetic.hook"),
        identity=SimpleNamespace(model_key="synthetic", sae_key="synthetic"),
    )
    snapshot = SimpleNamespace(
        codes=torch.tensor([0.0, -8.0, 2.0, 4.0]),
        reconstruction=torch.tensor([1.0, 2.0]),
    )
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    monkeypatch.setattr(operation, "_prompt", lambda cfg: "synthetic input")
    monkeypatch.setattr(saebase.api, "capture_features", lambda *args, **kwargs: snapshot)

    result = operation.execute(
        OmegaConf.create({"task": {"top_feature_num": requested, "prompt": "synthetic input"}})
    )

    assert [row["feature_id"] for row in result["top_features"]] == expected
    assert result["top_features"][0]["activation"] == -8.0
    assert result["feature_count"] == 4


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_invalid_count_is_rejected_before_model_loading(monkeypatch, value):
    def unexpected_load(cfg):
        pytest.fail("Invalid capture settings must not load a model")

    monkeypatch.setattr(operation, "_pair", unexpected_load)
    with pytest.raises(ValueError, match="task.top_feature_num"):
        operation.execute(OmegaConf.create({"task": {"top_feature_num": value}}))


def test_missing_count_preserves_default():
    assert operation.capture_top_feature_num(OmegaConf.create({"task": {}})) == 10


def test_observation_subset_preserves_parent_cohort_and_rejects_missing_ids():
    from saebase.datasets.selection import SampleSelection
    from tasks.capture.selection import observed_selection

    source = SampleSelection(
        [{"sample_id": "a"}, {"sample_id": "b"}], {"selected_count": 2, "records_sha256": "parent"}
    )
    assert observed_selection(source, None) is source
    selected = observed_selection(source, ["b"])
    assert list(selected) == [{"sample_id": "b"}]
    assert selected.metadata["selected_count"] == 1
    assert selected.metadata["parent_selection"] == source.metadata
    assert "records_sha256" not in selected.metadata
    assert source.metadata["selected_count"] == 2
    with pytest.raises(ValueError, match="absent"):
        observed_selection(source, ["c"])
    with pytest.raises(ValueError, match="unique"):
        observed_selection(source, ["a", "a"])
