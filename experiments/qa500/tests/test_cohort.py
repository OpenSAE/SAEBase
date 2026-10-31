"""Offline acceptance checks for the QA500 cohort and publication identity."""

from __future__ import annotations
import hashlib
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.qa500.cohort import (
    DATASETS,
    N_PER_DATASET,
    PROTOCOL_ID,
    normalize,
    render_prompt,
    digest,
    verify_dataset,
)
from saebase.integrations.neuronpedia import annotate


class QA500ContractTests(unittest.TestCase):
    def test_dataset_registry(self):
        self.assertEqual(len(DATASETS), 10)
        self.assertEqual(N_PER_DATASET, 500)

    def test_mmlu_answer_is_not_in_prompt(self):
        record = normalize(
            DATASETS["mmlu_pro"],
            {
                "question": "Which one?",
                "options": ["Wrong", "Right", "N/A"],
                "answer_index": 1,
                "category": "physics",
            },
            source_config=None,
            source_index=0,
        )
        self.assertEqual(record["answer"], "B")
        self.assertIn("B. Right", render_prompt(record))
        self.assertNotIn("Answer: B", render_prompt(record))
        self.assertEqual(record["answer_type"], "choice")

    def test_arc_label_remap(self):
        record = normalize(
            DATASETS["arc_challenge"],
            {
                "id": "arc-1",
                "question": "Which?",
                "choices": {"label": ["1", "2"], "text": ["first", "second"]},
                "answerKey": "2",
            },
            source_config="ARC-Challenge",
            source_index=0,
        )
        self.assertEqual(record["answer"], "B")

    def test_boolq_requires_real_boolean(self):
        with self.assertRaises(ValueError):
            normalize(
                DATASETS["boolq"],
                {"question": "Does it?", "answer": "True", "passage": "An article."},
                source_config=None,
                source_index=0,
            )

    def test_gsm8k_final_answer(self):
        record = normalize(
            DATASETS["gsm8k"],
            {"question": "Total?", "answer": "1+2 = 3\n#### 3"},
            source_config="main",
            source_index=0,
        )
        self.assertEqual(record["answer"], "3")
        self.assertNotIn("#### 3", render_prompt(record))

    def test_gpqa_shuffles_but_tracks_correct_option(self):
        record = normalize(
            DATASETS["gpqa_extended"],
            {
                "Question": "Q",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "a",
                "Incorrect Answer 2": "b",
                "Incorrect Answer 3": "c",
            },
            source_config="gpqa_extended",
            source_index=0,
        )
        self.assertEqual(
            next(c["text"] for c in record["choices"] if c["label"] == record["answer"]), "correct"
        )

    def test_neuronpedia_never_guesses_checkpoint(self):
        ref = {"site": {"model": "x", "source": "y", "sae_sha256": "a" * 64}}
        row = {"feature_id": 2, "model": "x", "source": "y", "description": "word"}
        result = annotate(
            2, site_id="site", sae_sha256="b" * 64, mapping=ref, annotations={"x/y/2": row}
        )
        self.assertEqual(result["explanation_status"], "checkpoint_mismatch")
        self.assertIsNone(result["explanation"])
        ok = annotate(
            2, site_id="site", sae_sha256="a" * 64, mapping=ref, annotations={"x/y/2": row}
        )
        self.assertEqual(ok["explanation"], "word")
        self.assertEqual(ok["explanation_status"], "source_reported")

    def test_verify_rejects_changed_prompt(self):
        records = []
        for i in range(500):
            row = normalize(
                DATASETS["boolq"],
                {"question": f"Q{i}?", "answer": i % 2 == 0, "passage": "Context"},
                source_config=None,
                source_index=i,
            )
            row["source_revision"] = "f" * 40
            records.append(row)
        payload = "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "boolq"
            root.mkdir()
            (root / "questions.jsonl").write_text(payload)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "n": 500,
                        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "sample_ids_sha256": digest([r["sample_id"] for r in records]),
                        "source_revision": "f" * 40,
                    }
                )
            )
            self.assertEqual(verify_dataset("boolq", tmp)["n"], 500)
            altered = (root / "questions.jsonl").read_text().replace("Q25", "Q25BAD")
            (root / "questions.jsonl").write_text(altered)
            with self.assertRaises(RuntimeError):
                verify_dataset("boolq", tmp)


if __name__ == "__main__":
    unittest.main()
