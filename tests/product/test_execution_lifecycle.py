"""Synthetic end-to-end task lifecycle tests; no network, real weights or GPU."""

import json
from dataclasses import replace

import pytest
from omegaconf import OmegaConf

from saebase.results import EvaluationResult, completion_summary
from saebase.runtime.experiment_paths import prepare_run, run_leaf, run_session
from saebase.runtime.experiment_results import compare_runs, find_reusable_runs, read_run
from saebase.runtime.train import execute_run
from tasks.sae_runtime import resolve_sae


def launch(root, task="capture"):
    return OmegaConf.create(
        {
            "model": {"name": "ext.toy.model" if task == "benchmark" else "pythia70m"},
            "dataset": {"name": "toy-data" if task == "benchmark" else "IOI"},
            "seed": 7,
            "paths": {"root": str(root)},
            "experiment": {"sub_name": "synthetic"},
            "runtime": {"mode": None, "clear_experiment_leaf": True},
            "task": {"name": task, "path": task, "execute": True, "device": "cpu", "resume": True},
        }
    )


@pytest.fixture
def external(monkeypatch, tmp_path):
    # Synthetic lifecycle fixtures provide a passed check; actual automatic gating
    # and prevention of task execution are covered by test_fixed_dataset_cohort.
    monkeypatch.setattr(
        "tasks.precheck.gate.automatic_precheck", lambda cfg: (None, "synthetic-precheck")
    )
    from saebase.extensions import register_extension, registry
    from tasks.benchmark.metrics import registry as metrics
    from tests.product.test_external_extensions import bundle, config

    for name in ("_MODELS", "_SAES", "_METRICS", "_METRIC_DIGESTS"):
        monkeypatch.setattr(registry, name, {})
    for name in ("_BUNDLES", "_DISCOVERED"):
        monkeypatch.setattr(registry, name, set())
    monkeypatch.setattr(metrics, "_METRICS", dict(metrics._METRICS))
    source = tmp_path / "synthetic_metric.py"
    source.write_text("def estimate(): return 0.5\n")
    calls = []
    original = bundle(source)
    factory = original.models[0].factory

    def counted(device):
        calls.append(device)
        return factory(device)

    provider = replace(original.models[0], factory=counted)
    register_extension(replace(original, models=(provider,)))
    protocol = config(source)
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(protocol.to_dict()))
    return original, protocol, path, calls


@pytest.mark.parametrize(
    "state, expected",
    [
        ("completed", 0),
        ("undefined", 0),
        ("blocked", 2),
        ("unsupported", 2),
        ("failed", 1),
    ],
)
def test_completion_and_exit_codes_agree(state, expected):
    summary = completion_summary([{"status": state, "reason": "synthetic reason"}])
    result = EvaluationResult(True, "benchmark", "benchmark", "toy", "toy", completion=summary)
    assert result.exit_code == expected
    assert result.ok is (expected == 0)
    assert result.status == "executed"


def test_unknown_and_missing_states_are_not_success():
    cfg = launch(".")
    for payload in ({"status": "typo"}, {}):
        with pytest.raises(ValueError, match="Unknown"):
            EvaluationResult.from_payload("capture", cfg, payload)
    with pytest.raises(ValueError, match="reason"):
        completion_summary([{"status": "undefined"}])


def test_invalid_configuration_preserves_existing_artifacts(monkeypatch, tmp_path):
    cfg = launch(tmp_path)
    cfg.task.top_feature_num = 0
    cfg.runtime.mode = "overwrite"
    folder = run_leaf(cfg)
    folder.mkdir(parents=True)
    marker = folder / "keep.txt"
    marker.write_text("previous result")
    monkeypatch.chdir(folder)
    with pytest.raises(ValueError, match="top_feature_num"):
        execute_run(cfg, folder)
    assert marker.read_text() == "previous result"


def test_ambiguous_layer_is_rejected_and_explicit_default_is_shared():
    from tasks.precheck.sae_checks import select_sites

    model = OmegaConf.create(
        {
            "name": "gemma2_2b_base",
            "sae_profiles": ["gemma2_2b_pt_l25_canonical"],
            "default_saes": ["gemma2_2b_pt_l25_canonical"],
        }
    )
    with pytest.raises(ValueError, match="multiple layers"):
        resolve_sae(model)
    model.default_layers = {"gemma2_2b_pt_l25_canonical": 25}
    selected = resolve_sae(model)
    assert selected.layer_source == "model.default_layers"
    assert select_sites(model, "default")[0]["layer"] == selected.layer


def test_external_capture_uses_actual_task_and_identity(external, tmp_path, monkeypatch):
    from tasks.capture.task import CaptureTask

    cfg = launch(tmp_path)
    cfg.model.name = "ext.toy.model"
    cfg.task.sae_key = "ext.toy.sae"
    cfg.task.prompt = "1 2 3"  # The synthetic tokenizer accepts integer tokens.
    cfg.task.top_feature_num = 3
    cfg.task.neuronpedia = {"enabled": False}  # This fixture has no published explanation source.
    monkeypatch.chdir(tmp_path)
    result = CaptureTask().run(cfg)
    assert result.exit_code == 0
    assert len(result.summary["top_features"]) == 3
    assert result.summary["selection"]["provider"] == "external"
    identity = result.summary["input_identity"]
    assert identity and len(identity["model_sha256"]) == 64
    assert identity["site"]["layer"] == 0
    from saebase.runtime.experiment_results import record_run

    record_run(tmp_path, cfg, result=result)
    candidates = find_reusable_runs(tmp_path, identity)
    assert len(candidates) == 1 and candidates[0]["reuse_authorized"] is False
    assert compare_runs(tmp_path, tmp_path)["supported"] is False


def test_declared_capture_only_provider_rejects_intervention_before_factory(external, tmp_path):
    from saebase.api import UnsupportedCapability
    from saebase.extensions import registry

    key = "ext.toy.model"
    registry._MODELS[key] = replace(registry._MODELS[key], capabilities=("capture",))
    cfg = launch(tmp_path, "intervene")
    cfg.model.name = key
    cfg.task.sae_key = "ext.toy.sae"
    with pytest.raises(UnsupportedCapability, match="patch"):
        resolve_sae(cfg.model, cfg.task.sae_key, operation="intervene")
    assert external[3] == []


def test_full_entry_resume_reuses_committed_ledger(external, tmp_path, monkeypatch):
    import saebase.runner as dispatcher
    from tasks.benchmark.execution import runner

    dispatched = []
    dispatch = dispatcher.run_task

    def traced(configuration, **kwargs):
        dispatched.append(configuration.task.name)
        return dispatch(configuration, **kwargs)

    monkeypatch.setattr(dispatcher, "run_task", traced)

    cfg = launch(tmp_path, "benchmark")
    cfg.task.benchmark_config = str(external[2])
    folder = run_leaf(cfg)
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    original = runner.execute

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic interruption after ledger commit")

    monkeypatch.setattr(runner, "execute", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        execute_run(cfg, folder)
    assert read_run(folder)["status"] == "failed"
    assert len(external[3]) == 1
    ledger = folder / "metrics" / external[1].run_id / "metrics.jsonl"
    before = ledger.read_bytes()
    monkeypatch.setattr(runner, "execute", original)
    cfg.runtime.mode = "resume"
    result = execute_run(cfg, folder)
    assert result.exit_code == 0 and result.completion["status"] == "complete"
    assert ledger.read_bytes() == before
    assert len(external[3]) == 1
    assert execute_run(cfg, folder).exit_code == 0
    assert len(external[3]) == 1
    assert len(read_run(folder)["resume_attempts"]) == 2
    assert dispatched == ["benchmark", "benchmark"]


def test_input_change_prevents_resume_without_altering_manifest(external, tmp_path, monkeypatch):
    cfg = launch(tmp_path, "benchmark")
    cfg.task.benchmark_config = str(external[2])
    folder = run_leaf(cfg)
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    execute_run(cfg, folder)
    before = (folder / "metadata/manifest.json").read_bytes()
    data = json.loads(external[2].read_text())
    data["seed"] += 1
    external[2].write_text(json.dumps(data))
    cfg.runtime.mode = "resume"
    with pytest.raises(ValueError, match="identity mismatch"):
        execute_run(cfg, folder)
    assert (folder / "metadata/manifest.json").read_bytes() == before


def test_inline_benchmark_executes_with_global_inputs_and_guards_sample_file_resume(
    external, tmp_path, monkeypatch
):
    cfg = launch(tmp_path, "benchmark")
    cfg.model.default_saes = ["ext.toy.sae"]
    raw = external[1].to_dict()
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(raw["dataset"]["samples"]))
    cfg.dataset.sample_file = str(samples)
    cfg.task.protocol = {
        key: value for key, value in raw.items() if key not in {"targets", "dataset", "name"}
    }
    folder = run_leaf(cfg)
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    result = execute_run(cfg, folder)
    assert result.exit_code == 0 and result.completion["status"] == "complete"
    assert result.model_key == cfg.model.name and result.prompt_source == cfg.dataset.name
    assert read_run(folder)["configuration"]["model"]["name"] == "ext.toy.model"
    cfg.runtime.mode = "resume"
    assert execute_run(cfg, folder).exit_code == 0 and len(external[3]) == 1
    before = (folder / "metadata/manifest.json").read_bytes()
    changed = raw["dataset"]["samples"]
    changed[0]["token_ids"] = [3, 2, 1]
    samples.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity mismatch"):
        execute_run(cfg, folder)
    assert (folder / "metadata/manifest.json").read_bytes() == before


def test_writer_lock_and_unsupported_resume(tmp_path):
    cfg = launch(tmp_path, "benchmark")
    folder = run_leaf(cfg)
    with run_session(cfg, folder):
        with pytest.raises(RuntimeError, match="active writer"):
            with run_session(cfg, folder):
                pytest.fail("A second writer acquired the lock")
    cfg.task.name = "capture"
    cfg.task.path = "capture"
    cfg.runtime.mode = "resume"
    with pytest.raises(ValueError, match="does not support"):
        prepare_run(cfg, run_leaf(cfg))


def test_benchmark_configuration_from_original_launch_directory(external, tmp_path, monkeypatch):
    from hydra.core.hydra_config import HydraConfig

    from tasks.benchmark.operation import configured_benchmark

    cfg = launch(tmp_path, "benchmark")
    cfg.task.benchmark_config = external[2].name
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    monkeypatch.setattr(HydraConfig, "initialized", staticmethod(lambda: True))
    monkeypatch.setattr(
        HydraConfig,
        "get",
        staticmethod(lambda: OmegaConf.create({"runtime": {"cwd": str(tmp_path)}})),
    )
    path, protocol = configured_benchmark(cfg)
    assert path == external[2]
    assert protocol.run_id == external[1].run_id


def test_benchmark_cannot_override_or_delete_global_component_selections(external, tmp_path):
    from saebase.runtime.cfg.config_validate import validate_hydra_cfg
    from tasks.benchmark.operation import configured_benchmark, resolved_protocol

    cfg = launch(tmp_path, "benchmark")
    cfg.model.name = "directory_label"
    cfg.dataset.name = "organizational_label"
    cfg.task.benchmark_config = str(external[2])
    before = OmegaConf.to_container(cfg)
    with pytest.raises((ValueError, KeyError), match="model|Model"):
        validate_hydra_cfg(cfg)
    assert OmegaConf.to_container(cfg) == before
    cfg.model.name = "ext.toy.model"
    cfg.dataset.name = "toy-data"
    validate_hydra_cfg(cfg)
    _, protocol = configured_benchmark(cfg)
    resolved = resolved_protocol(protocol, cfg)
    assert cfg.model.name == "ext.toy.model" and cfg.dataset.name == "toy-data"
    assert "launch_labels" not in resolved
    assert resolved["run_components"]["model"] == "ext.toy.model"
    assert resolved["targets"][0]["model_key"] == "ext.toy.model"
    cfg.task.sae_key = "different_sae"
    with pytest.raises(ValueError, match="conflicts"):
        validate_hydra_cfg(cfg)


def test_benchmark_cli_reports_incomplete_work_and_exit_two(tmp_path, monkeypatch, capsys):
    from copy import deepcopy

    from tasks.benchmark import cli
    from tasks.benchmark.execution import runner
    from tests.product.test_benchmark_execution import BASE

    raw = deepcopy(BASE)
    raw["targets"][0]["model_key"] = "pythia70m"
    raw["targets"][0]["sae_keys"] = ["pythia70m_resid_sm"]
    raw["metrics"] = ["core.l0"]
    raw["conditions"] = []
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(raw))
    row = {
        "model_id": "pythia70m",
        "sae_id": "pythia70m_resid_sm",
        "metric_id": "core.l0",
        "site": raw["targets"][0]["site"]
        if "site" in raw["targets"][0]
        else {"layer": raw["targets"][0]["layer"], "hook_name": raw["targets"][0]["hook_name"]},
        "status": "blocked",
        "reason": "synthetic missing resource",
    }
    monkeypatch.setattr(runner, "execute", lambda *a, **k: [row])
    monkeypatch.setattr(
        "tasks.precheck.gate.automatic_precheck", lambda cfg: (None, "synthetic-precheck")
    )
    assert cli.main(["--config", str(path), "--real", "--output", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["completion"]["counts"]["blocked"] == 1


def test_inconsistent_completion_counts_are_rejected():
    completion = completion_summary([{"status": "completed"}])
    completion["requested"] = 99
    with pytest.raises(ValueError, match="Inconsistent"):
        EvaluationResult(True, "capture", "capture", "toy", "toy", completion=completion)


def test_comparison_pairs_do_not_form_a_cartesian_product(tmp_path):
    from tasks.benchmark.results.metrics import compare_benchmark_runs, store_results

    first, second = tmp_path / "first", tmp_path / "second"
    for root, value in ((first, 1.0), (second, 2.0)):
        rows = [
            {
                "schema": "saebase.benchmark.metric.v1",
                "model_id": "toy",
                "sae_id": sae,
                "metric_id": "core.l0",
                "metric_version": "1",
                "protocol_id": "same",
                "dataset_id": "same",
                "split": "test",
                "seed": 0,
                "sample_unit": "token",
                "site": {"layer": 0},
                "status": "completed",
                "value": value,
                "metadata": {"data_sha256": "a" * 64, "protocol_configuration_sha256": "b" * 64},
            }
            for sae in ("a", "b")
        ]
        store_results(root / "metrics" / "run", rows, run_id="synthetic")
    compared = compare_benchmark_runs(first, second)
    assert len(compared["measurements"]) == 2
    assert all(row["left"]["sae_id"] == row["right"]["sae_id"] for row in compared["measurements"])
    explicit = compare_benchmark_runs(first, second, pairs=[(("toy", "a"), ("toy", "b"))])
    assert len(explicit["measurements"]) == 1
    assert explicit["measurements"][0]["delta"] == 1
    rows[0]["metadata"]["data_sha256"] = "c" * 64
    store_results(second / "metrics" / "run", rows, run_id="synthetic")
    changed = compare_benchmark_runs(first, second)
    assert changed["measurements"][0]["delta"] is None
    assert changed["measurements"][0]["comparable"] is False


def test_lab_forwards_task_summary_without_reinterpreting_it(tmp_path):
    from saebase.research_lab.backend.published_records import list_records, publish_record

    completion = completion_summary([{"status": "blocked", "reason": "synthetic fixture"}])
    record = {
        "schema": "opensae.wp12.p3.public_result.v1",
        "measured": True,
        "model_sha256": "a" * 64,
        "sae_sha256": "b" * 64,
        "completion": completion,
    }
    publish_record(
        "interventions", record, root=tmp_path, public_consent=True, rights_reviewed=True
    )
    item = list_records("interventions", root=tmp_path)["items"][0]
    assert item["completion"] == completion
    assert item["completion"]["status"] == "incomplete"
