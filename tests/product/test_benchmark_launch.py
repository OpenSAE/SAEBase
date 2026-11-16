"""One SAEBase master configuration and peer tasks; synthetic inputs, no weights."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from saebase.runner import run_task
from saebase.runtime.cfg.config_validate import validate_hydra_cfg
from saebase.runtime.experiment_paths import run_leaf, run_components
from tasks.benchmark.operation import configured_benchmark
from tests.product.test_benchmark_execution import BASE

ROOT = Path(__file__).resolve().parents[2]


def protocol():
    raw = deepcopy(BASE)
    raw["targets"] = [
        {
            "model_key": "pythia70m",
            "sae_keys": ["pythia70m_resid_sm"],
            "layer": 3,
            "hook_name": "blocks.3.hook_resid_post",
        }
    ]
    raw["dataset"]["id"] = "IOI"
    return raw


def launch(tmp_path, raw=None):
    raw = raw or protocol()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(raw))
    models = [{"name": target["model_key"]} for target in raw["targets"]]
    return OmegaConf.create(
        {
            "model": models[0] if len(models) == 1 else {"members": models},
            "dataset": {"name": raw.get("dataset", {}).get("id", "comparison-prompts")},
            "task": {
                "name": "benchmark",
                "path": "benchmark",
                "execute": False,
                "benchmark_config": str(path),
                "device": "cpu",
            },
            "paths": {"root": str(tmp_path)},
            "experiment": {"sub_name": "synthetic"},
            "seed": 42,
        }
    )


def test_single_master_and_no_task_specific_master():
    assert {path.name for path in (ROOT / "conf").glob("*.yaml")} == {"config.yaml"}


@pytest.mark.parametrize(
    "task",
    ["precheck", "capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"],
)
def test_every_task_uses_the_same_master_and_global_inputs(task):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
    before = OmegaConf.to_container(cfg)
    validate_hydra_cfg(cfg)
    assert cfg.task.name == task
    assert cfg.model.name == "gemma2_2b_base" and cfg.dataset.name == "IOI"
    assert OmegaConf.to_container(cfg) == before


def test_python_task_preserves_global_inputs(tmp_path):
    cfg = launch(tmp_path)
    before = OmegaConf.to_container(cfg)
    validate_hydra_cfg(cfg)
    result = run_task(cfg)
    assert OmegaConf.to_container(cfg) == before
    assert result.status == "configured_only" and result.model_key == "pythia70m"
    assert result.prompt_source == "IOI"
    assert run_leaf(cfg) == tmp_path / "experiment/IOI/pythia70m/benchmark/synthetic_42"
    assert not (tmp_path / "experiment").exists()


@pytest.mark.parametrize("entry", ["module", "public_cli"])
def test_hydra_uses_one_master_and_explicit_component_choices(tmp_path, entry):
    launch(tmp_path)
    command = (
        [sys.executable, "-m", "saebase.runtime.train"]
        if entry == "module"
        else [sys.executable, "-c", "from saebase.cli import main; main()"]
    )
    checked = subprocess.run(
        [
            *command,
            "model=pythia70m",
            "task=benchmark",
            "task.benchmark_config=protocol.json",
            "experiment.sub_name=synthetic",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    folder = tmp_path / "experiment/IOI/pythia70m/benchmark/synthetic_42"
    manifest = json.loads((folder / "metadata/manifest.json").read_text())
    assert manifest["run_components"] == {"dataset": "IOI", "model": "pythia70m"}
    assert manifest["configuration"]["model"]["name"] == "pythia70m"
    assert manifest["configuration"]["dataset"]["name"] == "IOI"


@pytest.mark.parametrize("group", ["model", "dataset"])
def test_task_file_cannot_override_global_selection(tmp_path, group):
    cfg = launch(tmp_path)
    cfg[group].name = "gemma2_2b_base" if group == "model" else "SimpleCopying"
    before = OmegaConf.to_container(cfg)
    with pytest.raises(ValueError, match=f"global {group}"):
        validate_hydra_cfg(cfg)
    assert OmegaConf.to_container(cfg) == before
    assert not (tmp_path / "experiment").exists()


def test_inline_method_parameters_use_global_model_and_dataset(tmp_path):
    cfg = launch(tmp_path)
    cfg.task.pop("benchmark_config")
    cfg.model.default_saes = ["pythia70m_resid_sm"]
    cfg.task.layer = 3
    cfg.dataset.samples = protocol()["dataset"]["samples"]
    cfg.task.protocol = {"metrics": ["core.l0"], "seed": 2}
    validate_hydra_cfg(cfg)
    path, scientific = configured_benchmark(cfg)
    assert path is None
    assert scientific.targets[0].model_key == cfg.model.name
    assert scientific.dataset_id == cfg.dataset.name
    assert scientific.samples[0].record.token_ids == (1, 2, 3)
    assert scientific.samples[0].record.split == "test"
    assert run_task(cfg).summary["resolved_protocol"]["protocol_seed"] == 2
    parameters = tmp_path / "method.json"
    parameters.write_text(json.dumps(OmegaConf.to_container(cfg.task.protocol)))
    cfg.task.pop("protocol")
    cfg.task.benchmark_config = str(parameters)
    _, from_file = configured_benchmark(cfg)
    assert from_file.to_dict() == scientific.to_dict()
    cfg.task.pop("benchmark_config")
    cfg.task.protocol = {"metrics": ["core.l0"]}
    cfg.task.protocol.targets = []
    with pytest.raises(ValueError, match="method parameters only"):
        validate_hydra_cfg(cfg)


def test_model_collection_is_global_and_never_a_task_dispatcher(tmp_path):
    raw = json.loads((ROOT / "conf/benchmark/model_comparison.json").read_text())
    cfg = launch(tmp_path, raw)
    before = OmegaConf.to_container(cfg)
    validate_hydra_cfg(cfg)
    result = run_task(cfg)
    assert OmegaConf.to_container(cfg) == before
    assert result.summary["resolved_protocol"]["run_components"] == run_components(cfg)
    assert run_components(cfg)["model"].startswith("model_set-")
    assert run_components(cfg)["dataset"] == "comparison-prompts"
    cfg.model.members = list(reversed(cfg.model.members))
    assert run_components(cfg)["model"] == result.model_key
    cfg.task.name = "capture"
    with pytest.raises(ValueError, match="does not support multiple models"):
        validate_hydra_cfg(cfg)


def test_ambiguous_global_model_selection_is_rejected(tmp_path):
    cfg = launch(tmp_path)
    cfg.model.members = [{"name": "gpt2small"}]
    with pytest.raises(ValueError, match="not both"):
        validate_hydra_cfg(cfg)


def test_task_file_cannot_silently_replace_global_samples(tmp_path):
    cfg = launch(tmp_path)
    cfg.dataset.samples = protocol()["dataset"]["samples"]
    cfg.dataset.samples[0].token_ids = [7, 8, 9]
    with pytest.raises(ValueError, match="global dataset input"):
        validate_hydra_cfg(cfg)


def test_catalog_presets_compose_and_allow_local_overrides():
    from conf import group_names, load_group
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from saebase.registry import list_registered_model_keys
    from saebase.plugin_bootstrap import ensure_plugins_loaded

    ensure_plugins_loaded()
    assert {load_group("model", key).model.name for key in group_names("model")} == set(
        list_registered_model_keys()
    )
    for group in ("model", "dataset", "sae", "prompt_source"):
        for key in group_names(group):
            selected = load_group(group, key)
            with initialize_config_module(version_base=None, config_module="conf"):
                prefix = "" if group in ("model", "dataset") else "+"
                cfg = compose(config_name="config", overrides=[f"{prefix}{group}={key}"])
            assert OmegaConf.to_container(cfg[group], resolve=False) == OmegaConf.to_container(
                selected[group], resolve=False
            )
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=pythia70m",
                "model.device=cpu",
                "dataset=ioi",
                "dataset.sample_count=3",
            ],
        )
    assert cfg.model.device == "cpu" and cfg.dataset.sample_count == 3
    assert load_group("model", "pythia70m").model.device == "auto"


def test_bounded_pair_cli_records_recovery_and_clears_resolved_failure(tmp_path, monkeypatch):
    from tasks.benchmark import cli, operation

    def missing(*args, **kwargs):
        raise FileNotFoundError("selected checkpoint path is absent")

    monkeypatch.setattr(operation, "run_pair_benchmark", missing)
    arguments = [
        "--pair-protocol",
        "cpu_pair",
        "--sae-key",
        "selected_sae",
        "--real",
        "--output",
        str(tmp_path),
    ]
    assert cli.main(arguments) == 1
    saved = json.loads((tmp_path / "pair_failure.json").read_text())
    assert saved["stage"] == "bounded_benchmark:cpu_pair"
    assert "selected checkpoint path" in saved["reason"]
    assert saved["repair"] and saved["impact"]
    assert saved["output"] == str(tmp_path.resolve())
    monkeypatch.setattr(operation, "run_pair_benchmark", lambda *a, **kw: {"status": "passed"})
    assert cli.main(arguments) == 0
    assert not (tmp_path / "pair_failure.json").exists()
