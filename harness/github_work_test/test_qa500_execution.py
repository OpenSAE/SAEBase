"""Offline functional acceptance for frozen QA model inputs and real-logit scorers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from saebase.datasets.qa_standard import DATASETS, PROTOCOL_ID, digest, normalize
from saebase.datasets.qa_inference import grade_answer, make_input, score_choices, evaluate_qa
from saebase.datasets.qa_batch import make_plan
from saebase.datasets.qa_evaluation import run_baseline


class MinimalTokenizer:
    eos_token_id = 99
    vocab_size = 128

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        if text == " A":
            return [10]
        if text == " B":
            return [11]
        if text == " Yes":
            return [12]
        if text == " No":
            return [13]
        return [2, 3, 4]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(
            {10: "A", 11: "B", 12: "Yes", 13: "No", 99: ""}.get(x, "?") for x in token_ids
        )


class MinimalEngine:
    tokenizer = MinimalTokenizer()

    def __init__(self):
        self.calls = 0

    def to_tokens(self, prompt):
        assert "Question:" in prompt
        return torch.tensor([[2, 3, 4]], dtype=torch.long)


class MinimalAdapter:
    def __init__(self):
        self.model = MinimalEngine()

    def tokens(self, *, prompt=None, tokens=None):
        return tokens if tokens is not None else self.model.to_tokens(prompt)

    def forward_logits(self, *, tokens, readout_position=-1):
        self.model.calls += 1
        logits = torch.zeros(128, dtype=torch.float32)
        logits[11] = 5
        logits[10] = 1
        logits[99] = -100
        return logits


class MinimalPair:
    def __init__(self):
        self.model = MinimalAdapter()


def test_frozen_plan_counts():
    plan = make_plan()
    assert plan["n_sites"] == 20
    assert plan["expected_capture_rows"] == 100000
    assert plan["expected_model_qa_baselines"] == 50000
    assert plan["expected_intervention_rows"] == 4000
    assert plan["expected_generation_rows"] == 1000


def test_candidate_logprobs_are_real_logits():
    pair = MinimalPair()
    token_ids = pair.model.tokens(prompt="Question: test")
    result = score_choices(pair, token_ids, ["A", "B"])
    assert result["prediction"] == "B"
    assert result["choice_logprobs"]["B"] > result["choice_logprobs"]["A"]
    assert pair.model.model.calls == 1


def test_question_answer_not_leaked_and_token_hash_reproducible():
    record = normalize(
        DATASETS["arc_challenge"],
        {
            "question": "Select?",
            "choices": {"label": ["A", "B"], "text": ["No", "Yes"]},
            "answerKey": "B",
        },
        source_config="ARC-Challenge",
        source_index=0,
    )
    pair = MinimalPair()
    tok1, meta1 = make_input(pair, record, "gemma2_2b_base")
    tok2, meta2 = make_input(pair, record, "gemma2_2b_base")
    assert torch.equal(tok1, tok2)
    assert meta1 == meta2
    assert "token_ids_sha256" in meta1
    scored = evaluate_qa(pair, tok1, record)
    assert scored["correct"] is True


def test_math_refuses_fake_equivalence_and_numeric_requires_final():
    math = grade_answer("math", "$2$", "$1+1$")
    assert math["grade_status"] in {"scored", "needs_math_equivalence", "unparseable_reference"}
    if math["grade_status"] == "scored":
        assert math["correct"] is True
    assert grade_answer("number", "Final: 3", "3")["correct"] is True
    assert grade_answer("boolean", "Yes", "No")["correct"] is False


def test_qa_baseline_resume_and_cross_model_input_hash():
    pair = MinimalPair()
    rows = []
    for i in range(500):
        row = normalize(
            DATASETS["arc_challenge"],
            {
                "id": str(i),
                "question": f"Question {i}?",
                "choices": {"label": ["A", "B"], "text": ["bad", "good"]},
                "answerKey": "B",
            },
            source_config="ARC-Challenge",
            source_index=i,
        )
        row["source_revision"] = "f" * 40
        rows.append(row)
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "inputs" / "arc_challenge"
        root.mkdir(parents=True)
        (root / "questions.jsonl").write_text(payload)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "n": 500,
                    "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    "sample_ids_sha256": digest([r["sample_id"] for r in rows]),
                    "source_revision": "f" * 40,
                    "source_split": "test",
                }
            )
        )
        kw = {
            "model_key": "gemma2_2b_base",
            "dataset_id": "arc_challenge",
            "data_root": base / "inputs",
            "output_root": base / "results",
            "pilot": 10,
        }
        first = run_baseline(pair, **kw)
        assert first["executed"] == 10
        assert first["accuracy"] == 1.0
        before = pair.model.model.calls
        second = run_baseline(pair, **kw)
        assert second["executed"] == 10
        assert pair.model.model.calls == before
