"""Offline device-policy regressions; mocked CUDA availability never loads weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from models import loading_policy as policy


@pytest.mark.parametrize("available,expected", [(True, "cuda"), (False, "cpu")])
@pytest.mark.parametrize(
    "task",
    ["precheck", "capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"],
)
def test_all_task_defaults_share_gpu_first_placement(monkeypatch, available, expected, task):
    monkeypatch.setattr(policy.torch.cuda, "is_available", lambda: available)
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
    assert cfg.task.device == cfg.model.device == "auto"
    assert policy.resolve_config_devices(cfg) == expected
    assert cfg.task.device == cfg.model.device == expected


def test_component_defaults_do_not_force_a_device():
    root = Path(__file__).resolve().parents[2]
    for directory, key in [("conf/models", "model"), ("conf/tasks", "task")]:
        for path in (root / directory).rglob("*.yaml"):
            cfg = OmegaConf.load(path)
            if key in cfg and "device" in cfg[key]:
                assert cfg[key].device == "auto", str(path)


@pytest.mark.parametrize("owner", ["runtime", "task", "model"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_explicit_device_propagates_to_automatic_peer(monkeypatch, owner, device):
    monkeypatch.setattr(
        policy.torch.cuda,
        "is_available",
        lambda: pytest.fail("Explicit placement must not probe CUDA"),
    )
    cfg = OmegaConf.create({"task": {"device": "auto"}, "model": {"device": "auto"}})
    cfg.runtime = {"device": "auto"}
    cfg[owner].device = device
    assert policy.resolve_config_devices(cfg) == device
    assert cfg.model.device == cfg.task.device == device


def test_conflicting_devices_fail_before_loading():
    cfg = OmegaConf.create({"model": {"device": "cpu"}, "task": {"device": "cuda"}})
    with pytest.raises(ValueError, match="devices conflict"):
        policy.resolve_config_devices(cfg)
    assert cfg.model.device == "cpu" and cfg.task.device == "cuda"


def test_generic_cuda_and_explicit_index_share_the_index():
    cfg = OmegaConf.create({"model": {"device": "cuda:1"}, "task": {"device": "cuda"}})
    assert policy.resolve_config_devices(cfg) == "cuda:1"
    assert cfg.model.device == cfg.task.device == "cuda:1"


@pytest.mark.parametrize("available,expected", [(True, "cuda"), (False, "cpu")])
def test_checkpoint_preparation_cli_uses_shared_default(monkeypatch, available, expected):
    from saebase.api.reference import parse_args

    monkeypatch.setattr(policy.torch.cuda, "is_available", lambda: available)
    assert parse_args([]).device == expected
    assert parse_args(["--device", "cpu"]).device == "cpu"


@pytest.mark.parametrize("available,expected", [(True, "cuda"), (False, "cpu")])
def test_direct_model_loading_uses_the_same_policy(monkeypatch, available, expected):
    from models import hooked_lm_common as common
    from models.model_adapter import ModelAdapter

    monkeypatch.setattr(policy.torch.cuda, "is_available", lambda: available)
    calls = []

    def loader(spec, **kwargs):
        calls.append(kwargs["device"])
        return SimpleNamespace(model=object())

    monkeypatch.setattr(common, "load_from_spec", loader)
    adapter = ModelAdapter(common.BaseModelSpec("synthetic", "synthetic/model", "synthetic"))
    adapter.load_model()
    adapter.construct(OmegaConf.create({"model": {"device": "auto"}}))
    assert calls == [expected, expected]


def test_dataset_preparation_receives_resolved_gpu(monkeypatch, tmp_path):
    from saebase.datasets.prompts import prepare_selected_dataset

    from saebase import registry

    monkeypatch.setattr(policy.torch.cuda, "is_available", lambda: True)
    calls = []

    def loader(cfg):
        calls.append(cfg.model.device)
        raise RuntimeError("synthetic stop before any resource loading")

    monkeypatch.setattr(
        registry, "get_model_plugin", lambda key: SimpleNamespace(load_model=loader)
    )
    cfg = OmegaConf.create(
        {
            "model": {"name": "gemma2_2b_base", "device": "auto"},
            "dataset": {"name": "IOI", "data_path": str(tmp_path / "IOI")},
            "paths": {"data_dir": str(tmp_path)},
            "seed": 42,
        }
    )
    with pytest.raises(RuntimeError, match="synthetic stop"):
        prepare_selected_dataset(cfg, n_generate=1, min_kept=1)
    assert calls == ["cuda"]
