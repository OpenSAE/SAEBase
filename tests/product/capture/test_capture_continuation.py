"""Synthetic local receipts for continuation and independent matrix failures."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from saebase.results import EvaluationResult, completion_summary
from saebase.runtime.experiment_results import inspect_run, record_run
from tasks.capture.matrix import CaptureMatrix

matrix = CaptureMatrix()


def config(root, name="IOI"):
    return OmegaConf.create(
        {
            "model": {"name": "gpt2small", "device": "cpu"},
            "dataset": {"name": name},
            "task": {"name": "capture", "execute": True, "top_feature_num": 10},
            "paths": {"root": str(root)},
            "experiment": {"sub_name": matrix.TAG},
            "seed": 42,
        }
    )


def receipt(root, *, annotations=False, legacy=False):
    ledger = root / "reports/capture_samples.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{}\n" * 3)
    complete = completion_summary([{"status": "completed"}] * 3, unit="sample")
    overall = completion_summary(
        [{"status": "blocked" if annotations else "completed"}] * 3, unit="sample"
    )
    result = {
        "status": "executed",
        "capture_format_version": 3,
        "selection_exhausted": True,
        "sample_count": 3,
        "ledger_rows": 3,
        "samples_path": str(ledger),
        "selection": {"sae_key": "synthetic"},
        "input_identity": {"selection": {"selected_count": 3}},
        "completion": overall,
    }
    if not legacy:
        result.update(
            stages={"capture": complete, "annotation": overall},
            required_artifacts=[
                {"path": "reports/capture_samples.jsonl", "bytes": ledger.stat().st_size}
            ],
        )
    return result


@pytest.mark.parametrize("annotations,expected", [(False, "skip"), (True, "annotations_pending")])
@pytest.mark.parametrize("legacy", [False, True])
def test_runtime_reads_task_policy_without_reading_ledger(
    tmp_path, monkeypatch, annotations, expected, legacy
):
    cfg = config(tmp_path)
    raw = receipt(tmp_path, annotations=annotations, legacy=legacy)
    record_run(tmp_path, cfg, result=EvaluationResult.from_payload("capture", cfg, raw))
    original = Path.open

    def no_ledger_read(path, *args, **kwargs):
        assert path.name != "capture_samples.jsonl"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_ledger_read)
    assert inspect_run(tmp_path, expected_cfg=cfg)["action"] == expected


@pytest.mark.parametrize("damage", ["missing", "size", "configuration", "manifest"])
def test_lightweight_inspection_rejects_incomplete_records(tmp_path, damage):
    cfg = config(tmp_path)
    raw = receipt(tmp_path)
    record_run(tmp_path, cfg, result=EvaluationResult.from_payload("capture", cfg, raw))
    ledger = tmp_path / "reports/capture_samples.jsonl"
    if damage == "missing":
        ledger.unlink()
    elif damage == "size":
        ledger.write_text("")
    elif damage == "configuration":
        cfg.task.top_feature_num = 3
    else:
        (tmp_path / "metadata/manifest.json").write_text("{")
    assert inspect_run(tmp_path, expected_cfg=cfg)["action"] == "retry"


@pytest.mark.parametrize("failed_name", ["IOI", "GSM8K"])
def test_any_dataset_preparation_failure_is_isolated(tmp_path, monkeypatch, failed_name):
    import saebase.datasets.selection as owner

    monkeypatch.setattr(matrix, "DATASETS", [(name, name) for name in ["IOI", "GSM8K", "BBH"]])
    called = []

    def prepare(cfg, **kwargs):
        called.append(cfg.name)
        if cfg.name == failed_name:
            raise RuntimeError("synthetic source failure")

    monkeypatch.setattr(owner, "select_dataset", prepare)
    pair = SimpleNamespace(
        sae=torch.nn.Linear(2, 2),
        model=SimpleNamespace(model=None),
        identity=SimpleNamespace(model_key="gpt2small", loaded_model_name="synthetic"),
    )
    errors = matrix.prepare_inputs(lambda alias, task: config(tmp_path, alias), pair)
    assert called == ["IOI", "GSM8K", "BBH"] and set(errors) == {failed_name}


@pytest.mark.parametrize("phase", ["config", "preparation", "execute"])
def test_failed_cell_does_not_prevent_next_and_retry_preserves_history(
    tmp_path, monkeypatch, phase
):
    import saebase.runtime.train as entry

    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "REPORT", tmp_path / "matrix")

    def factory(alias, task):
        if alias == "bad" and phase == "config":
            raise ValueError("synthetic config failure")
        return config(tmp_path, alias)

    def execute(cfg, leaf):
        if cfg.dataset.name == "bad" and phase == "execute":
            raise RuntimeError("synthetic execution failure")
        from saebase.runtime.experiment_paths import prepare_run

        prepare_run(cfg, leaf)
        result = EvaluationResult.from_payload("capture", cfg, receipt(leaf))
        record_run(leaf, cfg, result=result)
        return result

    monkeypatch.setattr(entry, "execute_run", execute)
    bad = matrix.run_cell(
        "bad",
        "bad",
        "gpt2small",
        "synthetic",
        0,
        factory,
        {"type": "RuntimeError", "message": "synthetic preparation failure"}
        if phase == "preparation"
        else None,
    )
    good = matrix.run_cell("good", "good", "gpt2small", "synthetic", 0, factory)
    assert bad["cell"]["tasks"]["capture"]["status"] == "failed"
    assert good["cell"]["tasks"]["capture"]["exit_code"] == 0
    had_execution = (matrix.member("bad", "gpt2small") / "metadata/manifest.json").is_file()
    # Both attempts share the canonical location. The execution archive preserves the failure.
    phase = "retry"
    retried = matrix.run_cell(
        "bad", "bad", "gpt2small", "synthetic", 0, lambda alias, task: config(tmp_path, alias)
    )
    from saebase.runtime.experiment_results import list_executions
    executions = list_executions(matrix.member("bad", "gpt2small"))
    assert executions[-1]["status"] == "executed"
    assert len(executions) == 1 + had_execution
    if had_execution:
        assert not executions[0]["current"]
    assert (
        matrix.run_cell(
            "bad", "bad", "gpt2small", "synthetic", 0, lambda alias, task: config(tmp_path, alias)
        )["action"]
        == "skip"
    )
    assert matrix.member("good", "gpt2small") != matrix.member("bad", "gpt2small")
    assert "run_dir" not in retried["cell"]["tasks"]["capture"]
    receipt_file = matrix.REPORT / "cells/bad--gpt2small.json"
    saved = json.loads(receipt_file.read_text())
    saved["tasks"]["capture"]["run_dir"] = "/unusable/old/result"
    receipt_file.write_text(json.dumps(saved))
    assert matrix.run_cell(
        "bad", "bad", "gpt2small", "synthetic", 0, lambda alias, task: config(tmp_path, alias)
    )["action"] == "skip"
    assert matrix.run_cell("good", "good", "gpt2small", "synthetic", 0, factory)["action"] == "skip"


@pytest.mark.parametrize("mode", ["iteration", "serialization", "sample", "annotation"])
def test_capture_totals_distinguish_sample_errors_from_run_errors(tmp_path, monkeypatch, mode):
    import tasks.capture.operation as operation
    import saebase.runtime.sae_operations as runtime
    import saebase.datasets.selection as owner
    from tasks.capture.batch import capture_dataset

    monkeypatch.chdir(tmp_path)
    pair = SimpleNamespace(
        identity=SimpleNamespace(
            model_key="synthetic",
            loaded_model_name="synthetic",
            sae_key="synthetic",
            layer=0,
            d_sae=4,
        ),
        site=SimpleNamespace(hook_name="synthetic"),
        model=SimpleNamespace(model=None),
    )
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    monkeypatch.setattr(runtime, "loaded_pair_session", lambda pair: nullcontext())
    monkeypatch.setattr(runtime, "pair_state_fingerprints", lambda pair: ("model", "sae"))

    def source():
        for index in range(5):
            if mode == "iteration" and index == 2:
                raise RuntimeError("synthetic iterator stopped")
            yield {"sample_id": str(index), "prompt": str(index)}

    monkeypatch.setattr(
        owner,
        "select_dataset",
        lambda *a, **kw: owner.SampleSelection(source(), {"selected_count": 5}),
    )

    def capture(cfg, *, pair, prompt, sample=None, model_answer=None):
        if mode == "sample" and prompt == "2":
            raise RuntimeError("synthetic sample failure")
        return {
            "completion": {"status": "incomplete" if mode == "annotation" else "complete"},
            "neuronpedia": {},
            "activation": float("nan") if mode == "serialization" and prompt == "2" else 1.0,
        }

    monkeypatch.setattr(operation, "_capture", capture)
    cfg = config(tmp_path, "synthetic")
    cfg.dataset.sample_count = 500
    raw = capture_dataset(cfg)
    result = EvaluationResult.from_payload("capture", cfg, raw)
    counts = result.completion["counts"]
    assert result.completion["requested"] == 5
    if mode in {"iteration", "serialization"}:
        assert counts["completed"] == 2 and counts["failed"] == 0
        assert result.completion["missing"] == 3 and result.status == "failed"
        assert raw["source_error"] is not None
    elif mode == "sample":
        assert (
            counts["completed"] == 4 and counts["failed"] == 1 and result.completion["missing"] == 0
        )
    else:
        assert raw["stages"]["capture"]["completed"] == 5
        assert raw["stages"]["capture"]["status"] == "complete"
        assert raw["stages"]["annotation"]["status"] == "incomplete"
    assert raw["ledger_rows"] == sum(counts.values())
