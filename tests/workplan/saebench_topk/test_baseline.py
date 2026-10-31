"""Offline identity and tensor-layout tests for the original SAEBench TopK SAE."""

from __future__ import annotations

import pytest
import torch

from saebase.sae.saebench_topk import (
    D_IN,
    D_SAE,
    FOLDER,
    HOOK,
    REPO,
    REVISION,
    remap_state_dict,
    validate_config,
    validate_source,
)


def _config() -> dict:
    return {
        "trainer": {
            "trainer_class": "TopKTrainer",
            "dict_class": "AutoEncoderTopK",
            "lm_name": "google/gemma-2-2b",
            "layer": 12,
            "submodule_name": "resid_post_layer_12",
            "activation_dim": D_IN,
            "dict_size": D_SAE,
            "k": 80,
        }
    }


def test_pinned_source_and_hook() -> None:
    validate_source(REPO, FOLDER, 12, HOOK, REVISION)
    validate_source(REPO, FOLDER, 12, HOOK, None)
    for repo, folder, layer, hook, revision in [
        (REPO, FOLDER, 13, HOOK, REVISION),
        (REPO, FOLDER, 12, "blocks.12.hook_resid_pre", REVISION),
        (REPO, FOLDER, 12, HOOK, "main"),
        ("other/repo", FOLDER, 12, HOOK, REVISION),
        (REPO, "../weights", 12, HOOK, REVISION),
    ]:
        with pytest.raises(ValueError, match="source identity"):
            validate_source(repo, folder, layer, hook, revision)


def test_config_is_strict() -> None:
    assert validate_config(_config())["k"] == 80
    for field, bad in (
        ("k", 40),
        ("layer", 13),
        ("activation_dim", 1024),
        ("trainer_class", "GatedSAETrainer"),
        ("lm_name", "other/model"),
    ):
        value = _config()
        value["trainer"][field] = bad
        with pytest.raises(ValueError, match="mismatch"):
            validate_config(value)


def test_weights_only_tensor_layout_and_topk() -> None:
    # CPU-only synthetic tensor fixture, no Hugging Face or weight downloads.
    raw = {
        "encoder.weight": torch.zeros(D_SAE, D_IN),
        "decoder.weight": torch.zeros(D_IN, D_SAE),
        "encoder.bias": torch.zeros(D_SAE),
        "bias": torch.zeros(D_IN),
        "k": torch.tensor(80),
        "threshold": torch.tensor(0.1),
    }
    converted = remap_state_dict(raw)
    assert tuple(converted["W_enc"].shape) == (D_IN, D_SAE)
    assert tuple(converted["W_dec"].shape) == (D_SAE, D_IN)
    assert "threshold" not in converted
    raw["k"] = torch.tensor(40)
    with pytest.raises(ValueError, match="shape/k"):
        remap_state_dict(raw)
