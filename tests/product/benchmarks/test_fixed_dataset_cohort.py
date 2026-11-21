"""Fixed-cohort and automatic execution-gate contracts; synthetic local records."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from saebase.datasets.sampling import sample_records, union_index
from saebase.datasets.selection import select_dataset, read_selection


@pytest.mark.parametrize(
    "total,expected", [(164, 164), (400, 400), (450, 450), (501, 500), (10000, 500)]
)
def test_fixed_count_uniform_reproducible(total, expected):
    def get(i):
        return {
            "sample_id": str(i),
            "prompt": str(i),
            "split": "train" if i < total - 1 else "test",
        }

    rows = sample_records(total, get, 500, 42)
    assert len(rows) == expected
    assert len({r["sample_id"] for r in rows}) == expected
    assert rows == sample_records(total, get, 500, 42)
    if total > 500:
        assert rows != sample_records(total, get, 500, 43)
        assert max(int(r["sample_id"]) for r in rows) > 499


def test_split_union_uses_every_record():
    total, get = union_index({"train": [1, 2], "validation": [3], "test": [4, 5]})
    assert total == 5
    assert [get(i) for i in range(total)] == [
        ("train", 0, 1),
        ("train", 1, 2),
        ("validation", 0, 3),
        ("test", 0, 4),
        ("test", 1, 5),
    ]


def test_persisted_cohort_shared_by_precheck_and_tasks(tmp_path, monkeypatch):
    import saebase.registry as registry
    from saebase.datasets.selection import SampleSelection
    from saebase.datasets.prompts import read_prompts, prepare_selected_dataset
    from tasks.precheck.dataset_checks import check_dataset_artifact

    rows = [
        {"sample_id": str(i), "prompt": "same valid text", "split": "test" if i % 2 else "train"}
        for i in range(500)
    ]
    calls = []

    def choose(*a, **kw):
        calls.append(kw)
        return SampleSelection(rows, {"population_count": 1000})

    monkeypatch.setattr(
        registry, "get_dataset_plugin", lambda _: SimpleNamespace(select_samples=choose)
    )
    cfg = OmegaConf.create(
        {
            "model": {"name": "synthetic"},
            "dataset": {
                "name": "SyntheticNatural",
                "data_path": str(tmp_path),
                "sample_count": 500,
                "sampling_seed": 42,
            },
            "task": {"device": "cpu"},
            "paths": {"data_dir": str(tmp_path)},
        }
    )
    select_dataset(cfg.dataset, model_key="model1")
    assert len(read_prompts(cfg.dataset, model_key="model2")) == 500
    prepared = prepare_selected_dataset(cfg, n_generate=32, min_kept=1)
    assert prepared["records"] == 500
    check = check_dataset_artifact(
        dataset_key="SyntheticNatural",
        model_key="model2",
        data_root=tmp_path,
        dataset_cfg=cfg.dataset,
        max_records=2,
    )
    assert check["status"] == "PASS" and check["observed"] == 2 and check["population_count"] == 500
    assert len(calls) == 1
    path = tmp_path / "_shared" / "selections"
    next(path.glob("*/records.jsonl")).write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        read_selection(cfg.dataset, "model1")


@pytest.mark.parametrize(
    "name", ["capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"]
)
@pytest.mark.parametrize("passed", [True, False])
def test_public_dispatch_always_gates_real_task(name, passed, monkeypatch):
    import saebase.registry as registry
    import tasks.precheck.gate as gate
    from saebase.runner import run_task
    from saebase.results import EvaluationResult

    events = []

    def result(ok):
        return EvaluationResult(
            ok=ok,
            task=name,
            evaluation=name,
            model_key="synthetic",
            prompt_source="synthetic",
            status="executed" if ok else "blocked",
        )

    class Task:
        def validate_cfg(self, cfg):
            events.append("validate")

        def run(self, cfg):
            events.append("run")
            return result(True)

    def check(cfg):
        events.append("precheck")
        return (None if passed else result(False)), "synthetic-receipt"

    monkeypatch.setattr(registry, "get_task_plugin", lambda _: SimpleNamespace(task_class=Task))
    monkeypatch.setattr(gate, "automatic_precheck", check)
    cfg = OmegaConf.create(
        {
            "model": {"name": "synthetic", "device": "cpu"},
            "dataset": {"name": "synthetic"},
            "task": {"name": name, "execute": True, "device": "cpu"},
        }
    )
    outcome = run_task(cfg)
    assert events == (["validate", "precheck", "run"] if passed else ["validate", "precheck"])
    assert outcome.ok == passed


def test_batch_capture_keeps_every_row_and_partial_failure(tmp_path, monkeypatch):
    from contextlib import nullcontext
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
    monkeypatch.setattr(runtime, "pair_state_fingerprints", lambda pair: ("model-sha", "sae-sha"))
    rows = [{"sample_id": str(i), "prompt": str(i), "split": "test"} for i in range(3)]
    monkeypatch.setattr(
        owner, "select_dataset", lambda *a, **kw: owner.SampleSelection(rows, {"selected_count": 3})
    )
    calls = []

    monkeypatch.setattr(
        "tasks.capture.model_answer.answers",
        lambda cfg, pair, prompts: [{"status": "completed", "text": prompt} for prompt in prompts],
    )

    def capture(cfg, *, pair, prompt, sample=None, model_answer):
        calls.append(prompt)
        if prompt == "1":
            raise RuntimeError("synthetic middle failure")
        return {
            "completion": {"status": "complete"},
            "neuronpedia": {},
            "top_features": [{"feature_id": 1, "activation": 2.0}],
        }

    monkeypatch.setattr(operation, "_capture", capture)
    cfg = OmegaConf.create({"dataset": {"name": "synthetic", "sample_count": 500}, "seed": 42})
    result = capture_dataset(cfg)
    assert calls == ["0", "1", "2"]
    assert result["sample_count"] == 3 and result["selection_exhausted"]
    assert result["completion"]["counts"] == {
        "completed": 2,
        "undefined": 0,
        "blocked": 0,
        "unsupported": 0,
        "failed": 1,
    }
    import json

    entries = [
        json.loads(line)
        for line in (tmp_path / "reports/capture_samples.jsonl").read_text().splitlines()
    ]
    assert [entry["status"] for entry in entries] == ["completed", "failed", "completed"]


def test_synthetic_filter_stops_only_after_requested_eligible_count():
    import torch
    from saebase.datasets.indirect_object_identification.tokenization import filter_argmax

    class Model:
        calls = 0

        def __call__(self, prompt):
            self.calls += 1
            return torch.tensor([[[0.0, 1.0]]])

    records = [{"prompt": str(i), "tok_io": int(i % 2 == 1)} for i in range(20)]
    model = Model()
    kept = filter_argmax(records, model, max_kept=3)
    assert [r["prompt"] for r in kept] == ["1", "3", "5"]
    assert model.calls == 6
    assert len(filter_argmax(records, Model())) == 10


def test_synthetic_generation_retries_then_retains_exact_count(tmp_path, monkeypatch):
    import json
    import saebase.datasets.indirect_object_identification.prepare as owner

    calls = []

    def prepare(model_key, model, **parameters):
        calls.append(parameters)
        if len(calls) == 1:
            raise RuntimeError("kept 400 < min_kept=500")
        path = tmp_path / "generated"
        path.mkdir()
        (path / "manifest.json").write_text("{}")
        (path / "prompts.jsonl").write_text(
            "".join(json.dumps({"sample_id": str(i), "prompt": str(i)}) + "\n" for i in range(500))
        )
        return path

    monkeypatch.setattr(owner, "prepare_for_model", prepare)
    cfg = OmegaConf.create({"sample_count": 500, "max_candidates": 4000})
    result = owner.select_samples(
        cfg, model_key="synthetic", model=object(), loaded_model_name="synthetic"
    )
    assert len(list(result)) == 500
    assert [c["n_generate"] for c in calls] == [2000, 4000]
    assert all(c["stop_after"] == 500 and c["min_kept"] == 500 for c in calls)


def test_default_datamodule_retains_all_original_splits(monkeypatch):
    from saebase.registry import get_dataset_plugin
    import saebase.datasets.prompts as owner

    rows = [
        {"prompt": "synthetic", "sample_id": str(i), "split": split}
        for i, split in enumerate(["train", "validation", "test", "unspecified"])
    ]
    monkeypatch.setattr(owner, "read_prompts", lambda *a, **kw: rows)
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "GSM8K", "sample_count": 500},
            "model": {"name": "synthetic"},
            "task": {"name": "capture"},
        }
    )
    dm = SimpleNamespace()
    get_dataset_plugin("GSM8K").setup(dm, cfg)
    assert dm.observation_set == rows and dm.test_set == rows
    assert dm.train_set == [] and dm.val_set == []
