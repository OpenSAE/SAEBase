"""TrainPlan dtype and local-dataset contract. No GPU and no weight download."""

from __future__ import annotations

from pathlib import Path

import pytest

from tasks.LLM.train.backend.training import TrainPlan, _plan_identity, run_training


def _plan(tmp_path: Path, **overrides) -> TrainPlan:
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    fields = dict(
        backend="saelens",
        model_path=str(model),
        dataset_path=str(data),
        model_revision="a" * 40,
        dataset_revision="b" * 40,
        hookpoint="blocks.0.hook_resid_post",
        d_in=8,
        d_sae=16,
        k=4,
        architecture="topk",
        max_examples=32,
        steps=2,
        batch_size=16,
        ctx_len=16,
        run_name="contract",
        device="cpu",
    )
    fields.update(overrides)
    return TrainPlan(**fields)


def test_defaults_match_historical_float32_plan(tmp_path: Path):
    plan = _plan(tmp_path)
    plan.validate()
    assert plan.model_dtype == "float32"
    assert plan.sae_dtype == "float32"
    assert plan.store_batch_size_prompts is None
    assert plan.n_batches_in_buffer is None


def test_invalid_dtype_and_sparsify_precision_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="model_dtype"):
        _plan(tmp_path, model_dtype="int8").validate()
    with pytest.raises(ValueError, match="Sparsify"):
        _plan(tmp_path, backend="sparsify", model_dtype="float16").validate()


def test_jsonl_file_is_a_dataset_and_missing_path_is_not(tmp_path: Path):
    data = tmp_path / "texts.jsonl"
    data.write_text('{"text": "hello"}\n', encoding="utf-8")
    _plan(tmp_path, dataset_path=str(data)).validate()
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="dataset_path"):
        _plan(tmp_path, dataset_path=str(missing)).validate()


def test_unapproved_training_is_refused(tmp_path: Path):
    with pytest.raises(PermissionError, match="authorized"):
        run_training(_plan(tmp_path), output=tmp_path / "out", approved=False)


def test_old_plan_json_without_dtype_fields_still_matches(tmp_path: Path):
    plan = _plan(tmp_path)
    historical = {
        "backend": plan.backend,
        "model_path": plan.model_path,
        "dataset_path": plan.dataset_path,
        "model_revision": plan.model_revision,
        "dataset_revision": plan.dataset_revision,
        "hookpoint": plan.hookpoint,
        "d_in": plan.d_in,
        "d_sae": plan.d_sae,
        "k": plan.k,
        "architecture": plan.architecture,
        "max_examples": plan.max_examples,
        "steps": plan.steps,
        "batch_size": plan.batch_size,
        "ctx_len": plan.ctx_len,
        "run_name": plan.run_name,
        "device": plan.device,
        "distributed_gpus": plan.distributed_gpus,
    }
    assert _plan_identity(historical) == _plan_identity(plan.__dict__ | {"resume_checkpoint": None})
