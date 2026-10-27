"""Circuit-tracer option forwarding. Does not launch the worker or load a .pt."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.LLM.circuits.backend.circuits import import_native_graph_file, run_native_circuit_tracer
from tasks.LLM.circuits.operation import execute


def test_unapproved_circuit_is_refused(tmp_path: Path):
    with pytest.raises(PermissionError):
        run_native_circuit_tracer(
            prompt="The capital of France is",
            transcoder_set="gemma",
            output=tmp_path / "graph",
            slug="gpuacc-france",
            approved=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"offload": "gpu"}, "offload"),
        ({"dtype": "int8"}, "dtype"),
        ({"max_feature_nodes": 0}, "Feature-node"),
        ({"batch_size": 0}, "batch"),
        ({"timeout_seconds": 10}, "budget"),
    ],
)
def test_illegal_circuit_options_fail_before_the_worker(tmp_path: Path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_native_circuit_tracer(
            prompt="The capital of France is",
            transcoder_set="gemma",
            output=tmp_path / "graph",
            slug="gpuacc-france",
            approved=True,
            **kwargs,
        )


def test_operation_forwards_memory_options_and_keeps_defaults(monkeypatch, tmp_path):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"status": "native_raw_graph_generated_not_normalized"}

    monkeypatch.setattr(
        "tasks.LLM.circuits.backend.circuits.run_native_circuit_tracer", fake
    )
    cfg = SimpleNamespace(
        task={
            "approved": True,
            "prompt": "The capital of France is",
            "transcoder_set": "gemma",
            "slug": "gpuacc-france",
            "offload": "cpu",
            "dtype": "float16",
            "batch_size": 1,
            "max_feature_nodes": 128,
            "lazy_encoder": True,
        }
    )
    monkeypatch.chdir(tmp_path)
    result = execute(cfg)
    assert result["status"] == "executed"
    assert captured["offload"] == "cpu"
    assert captured["dtype"] == "float16"
    assert captured["batch_size"] == 1
    assert captured["max_feature_nodes"] == 128
    assert captured["lazy_encoder"] is True
    assert captured["approved"] is True


def test_operation_defaults_do_not_invent_options(monkeypatch, tmp_path):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"status": "native_raw_graph_generated_not_normalized"}

    monkeypatch.setattr(
        "tasks.LLM.circuits.backend.circuits.run_native_circuit_tracer", fake
    )
    monkeypatch.chdir(tmp_path)
    execute(
        SimpleNamespace(
            task={"approved": True, "prompt": "The capital of France is", "slug": "gpuacc-france"}
        )
    )
    assert captured["offload"] is None
    assert captured["dtype"] is None
    assert captured["batch_size"] is None
    assert captured["max_feature_nodes"] is None
    assert captured["lazy_encoder"] is False


def test_circuit_tracer_0_5_error_node_is_not_silently_converted(tmp_path: Path):
    """0.5.0 writes ``{layer}_{other}_{pos}`` error ids, not the pinned ``layer_-1_pos`` form."""
    source = tmp_path / "graph.json"
    source.write_text(
        """
        {
          "metadata": {
            "schema_version": 1,
            "scan": "mwhanna/gemma-scope-transcoders",
            "prompt_tokens": ["<bos>", "The"]
          },
          "nodes": [
            {
              "node_id": "0_1_2",
              "feature_type": "mlp reconstruction error",
              "ctx_idx": 2,
              "layer": "1"
            }
          ],
          "links": []
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reconstruction-error"):
        import_native_graph_file(
            source,
            model_sha256="a" * 64,
            transcoder_sha256="b" * 64,
            original_tool_commit="c" * 40,
        )
