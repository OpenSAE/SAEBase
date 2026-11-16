"""End-to-end local candidate and held-out scoring, no external providers."""

from __future__ import annotations

from collections import UserDict
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.auto_interp_local import LocalAutoInterpreter  # noqa: E402


class ToyTokenizer:
    eos_token_id = 0

    def __call__(self, text, *, return_tensors, truncation):
        assert return_tensors == "pt"
        assert truncation is False
        if text.startswith("You are interpreting"):
            index = 1
        elif "\nText: Positive" in text:
            index = 2
        else:
            index = 3
        # HF Transformers returns a BatchEncoding/UserDict, not a plain dict.
        return UserDict({"input_ids": torch.tensor([[index]])})

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return {
            11: "A pattern involving positive expressions.",
            12: "YES",
            13: "NO",
        }[int(tokens[0])]


class ToyModel:
    def __init__(self):
        self.generate_calls = 0

    def generate(
        self,
        *,
        input_ids,
        max_new_tokens,
        do_sample,
        num_beams,
        pad_token_id,
    ):
        assert do_sample is False and num_beams == 1 and pad_token_id == 0
        assert 1 <= max_new_tokens <= 100
        self.generate_calls += 1
        extra = {1: 11, 2: 12, 3: 13}[int(input_ids[0, 0])]
        return torch.cat([input_ids, torch.tensor([[extra]])], dim=-1)


def test_local_interpretation_executes_model_and_heldout_prediction():
    model = ToyModel()
    worker = LocalAutoInterpreter(
        model,
        ToyTokenizer(),
        model_name="/local/hf-source",
        model_sha256="a" * 64,
        device="cpu",
    )
    discovery = [{"group_id": "d1", "text": "Positive observation", "activation": 1.5}]
    heldout = [
        {"group_id": "e1", "text": "Positive heldout", "activation": 0.2},
        {"group_id": "e2", "text": "Negative heldout", "activation": 0.0},
    ]
    outcome = worker.interpret(
        site_id="source:sae:L3",
        feature_id=12,
        discovery=discovery,
        heldout=heldout,
    )
    assert outcome["description"] == "A pattern involving positive expressions."
    assert outcome["heldout_score"] == 1.0
    assert outcome["approved"] is False
    assert outcome["private_unpublished"] is True
    assert outcome["compute_mode"] == "local_original_model_forward"
    assert outcome["generator_state_sha256"] == "a" * 64
    assert model.generate_calls == 3


def test_local_interpreter_rejects_missing_checkpoint_and_invalid_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalAutoInterpreter.from_local_snapshot(tmp_path / "missing")
    with pytest.raises(ValueError, match="source identity"):
        LocalAutoInterpreter(
            ToyModel(),
            ToyTokenizer(),
            model_name="unknown",
            model_sha256="bad",
            device="cpu",
        )
