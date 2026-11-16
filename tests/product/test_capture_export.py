"""Purpose and main functions: Check capture publication through the actual Parquet reader.

Scope: Local synthetic native/whiten observations, saved explanation updates and normalization artifacts.
"""

import json

import numpy as np
import pyarrow.parquet as pq

from tasks.capture.export import export_matrices


def test_saved_capture_modes_share_explanations_and_keep_zero_ranked_values(tmp_path):
    def description(label):
        return {
            "id": label,
            "description": label,
            "method": "recorded",
            "explainer_model": "example",
        }

    def available(label):
        return {
            "status": "available",
            "explanations": [description(label)],
            "url": "https://www.neuronpedia.org/example/3/1",
        }

    missing = {"status": "unexplained", "explanations": []}
    matrices = []
    for mode in ("non_whiten", "whiten"):
        run = tmp_path / mode
        (run / "metadata").mkdir(parents=True)
        (run / "reports").mkdir()
        native = {"feature_id": 1, "activation": 2.0, "neuronpedia": available("first")}
        second = {"feature_id": 2, "activation": 3.0, "neuronpedia": missing}
        top = (
            second
            if mode == "non_whiten"
            else {
                "feature_id": 1,
                "activation": 0.0,
                "zscore": -2.0,
                "mean": 1.0,
                "std": 0.5,
                "whiten_eligible": True,
                "zero_native_activation": True,
                "neuronpedia": missing,
            }
        )
        result = {
            "model_key": "example",
            "sae_key": "dictionary",
            "layer": 3,
            "site": "blocks.3.resid",
            "prompt_sha256": "p" * 64,
            "input_identity": {
                "protocol_id": "saebase.capture.v1",
                "model_sha256": "m" * 64,
                "sae_sha256": "s" * 64,
            },
            "active_features": [native, second] if mode == "non_whiten" else [second],
            "top_features": [top],
        }
        if mode == "whiten":
            result["mode"] = mode
            np.savez(run / "metadata/capture_normalization.npz", mean=[1.0, 2.0], std=[0.5, 1.0])
            (run / "metadata/capture_normalization.json").write_text(
                json.dumps(
                    {
                        "identity": {
                            "content_sha256": "n" * 64,
                            "protocol_id": "zscore",
                            "cohort_sha256": "c" * 64,
                            "token_count": 2,
                        },
                        "prompt_count": 1,
                        "feature_count": 2,
                        "valid_feature_count": 2,
                    }
                )
            )
        (run / "reports/capture_samples.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": "sample:1",
                    "split": "test",
                    "source_index": 0,
                    "status": "completed",
                    "result": result,
                }
            )
            + "\n"
        )
        if mode == "non_whiten":
            (run / "reports/capture_explanations.json").write_text(
                json.dumps(
                    {
                        "annotation": {"status": "complete", "export_sha256": "e" * 64},
                        "features": {"2": available("second")},
                    }
                )
            )
        (run / "metadata/manifest.json").write_text(
            json.dumps(
                {"status": "executed", "source": {"commit": "c" * 40}, "result": {"summary": {}}}
            )
        )
        matrix = tmp_path / (mode + ".json")
        matrix.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "dataset": "Example",
                            "model": "example",
                            "sae": "dictionary",
                            "layer": 3,
                            "tasks": {"capture": {"run_dir": str(run)}},
                        }
                    ]
                }
            )
        )
        matrices.append((mode, matrix))
    output = tmp_path / "export"
    manifest = export_matrices(matrices, output)
    assert manifest["features"] == 2
    features = pq.read_table(next((output / "features").glob("*.parquet"))).to_pylist()
    assert {row["explanations"][0]["description"] for row in features} == {"first", "second"}
    white = pq.read_table(output / "activations/whiten/example/Example.parquet").to_pylist()
    selected = next(row for row in white if row["is_top"])
    assert selected["activation"] == 0 and selected["zscore"] == -2
    assert selected["is_active"] is False and selected["zero_native_activation"] is True
    assert manifest["totals"]["whiten"]["samples_with_top_explanations"] == 1
    assert manifest["totals"]["non_whiten"]["samples_with_top_explanations"] == 1
    assert len(pq.read_table(output / "normalization.parquet")) == 1
