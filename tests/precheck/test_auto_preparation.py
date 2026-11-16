"""Offline preparation regressions; Hub/pip responses and models are synthetic fixtures."""

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from types import ModuleType

from omegaconf import OmegaConf
import pytest

from saebase.setup import assets


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Hub protocol stub; the offline CI job does not install analysis extras."""
    module = ModuleType("huggingface_hub")
    module.HfApi = None
    module.snapshot_download = None
    constants = ModuleType("huggingface_hub.constants")
    constants.HF_HUB_CACHE = str(tmp_path / "hub-cache")
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    return module


def test_precheck_defaults_prepare_and_do_not_add_compute_permission_flags():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "conf/task/precheck.yaml")
    assert (
        cfg.task.auto_setup
        and cfg.task.allow_network
        and cfg.task.allow_install
        and cfg.task.allow_download
    )
    assert cfg.task.max_download_bytes > 0
    assert not {"allow_compute", "allow_gpu", "allow_native"} & set(cfg.task)


def test_installed_dependency_preparation_uses_distribution_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(assets.importlib.metadata, "version", lambda name: "1.2.3")
    assert assets._installation_target(tmp_path) == ["opensae-saebase[analysis]==1.2.3"]
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    assert assets._installation_target(tmp_path) == ["-e", ".[analysis]"]


def test_install_timeout_preserves_partial_output(monkeypatch, tmp_path):
    monkeypatch.setattr(assets.importlib.util, "find_spec", lambda name: None)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("synthetic pip", 1, output=b"partial", stderr=b"reason")

    monkeypatch.setattr(assets.subprocess, "run", timeout)
    result = assets.prepare_environment(required_modules=("transformer_lens",), log_dir=tmp_path)
    assert result["status"] == "BLOCKED" and result["code"] == "BLOCKED_DEPENDENCY"
    assert (tmp_path / "pip.stdout.log").read_text() == "partial"
    assert (tmp_path / "pip.stderr.log").read_text() == "reason"


def test_install_attempt_precedes_blocked_and_logs_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(assets.importlib.util, "find_spec", lambda name: None)
    commands = []

    def pip_failure(command, **kwargs):
        commands.append(command)
        assert "torch==" in Path(command[command.index("--constraint") + 1]).read_text()
        return subprocess.CompletedProcess(
            command, 1, "synthetic stdout", "synthetic resolver conflict"
        )

    monkeypatch.setattr(assets.subprocess, "run", pip_failure)
    result = assets.prepare_environment(required_modules=("transformer_lens",), log_dir=tmp_path)
    assert commands and commands[0][1:4] == ["-m", "pip", "install"]
    assert result["status"] == "BLOCKED" and result["code"] == "BLOCKED_DEPENDENCY"
    assert (tmp_path / "pip.stderr.log").read_text() == "synthetic resolver conflict"
    assert result["events"][-1]["exit_code"] == 1


def test_install_success_requires_a_fresh_import(monkeypatch, tmp_path):
    monkeypatch.setattr(assets.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        assets.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", "")
    )
    monkeypatch.setattr(
        assets,
        "_import_check",
        lambda names: subprocess.CompletedProcess([], 1, "", "synthetic import failure"),
    )
    result = assets.prepare_environment(required_modules=("transformer_lens",), log_dir=tmp_path)
    assert result["status"] == "BLOCKED" and "import failure" in result["reason"]


def test_partial_cache_downloads_only_missing_selected_files_at_pinned_revision(
    monkeypatch, tmp_path, hub
):
    sha = "a" * 40
    cache = tmp_path / sha
    cache.mkdir()
    (cache / "config.json").write_text("{}")
    (cache / "tokenizer.json").write_text("{}")
    calls = []
    info = SimpleNamespace(
        sha=sha,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=2),
            SimpleNamespace(rfilename="tokenizer.json", size=2),
            SimpleNamespace(rfilename="model.safetensors", size=3),
            SimpleNamespace(rfilename="pytorch_model.bin", size=10**12),
        ],
    )

    class Api:
        def model_info(self, *args, **kwargs):
            calls.append("metadata")
            return info

    def snapshot(**kwargs):
        if not kwargs.get("local_files_only"):
            assert kwargs["revision"] == sha
            assert set(kwargs["allow_patterns"]) == {
                "config.json",
                "tokenizer.json",
                "model.safetensors",
            }
            (cache / "model.safetensors").write_bytes(b"toy")
            calls.append("download")
        return str(cache)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    request = [{"kind": "model", "repo_id": "synthetic/model"}]
    result = assets.prepare_environment(required_modules=(), assets=request, max_download_bytes=3)
    assert result["status"] == "READY" and calls == ["metadata", "download"]
    assert result["prepared_assets"][0]["download_bytes"] == 3
    assert result["scientific_test"] == "NOT_RUN"
    assert (
        assets.prepare_environment(required_modules=(), assets=request, allow_network=False)[
            "status"
        ]
        == "READY"
    )
    assert calls == ["metadata", "download"]


def test_download_error_is_blocked_after_attempt_and_keeps_exact_source(monkeypatch, hub):
    def absent(**kwargs):
        raise FileNotFoundError("synthetic uncached asset")

    class Api:
        def model_info(self, repo, **kwargs):
            assert repo == "synthetic/original"
            raise OSError("synthetic 403 source access denied")

    monkeypatch.setattr(hub, "snapshot_download", absent)
    monkeypatch.setattr(hub, "HfApi", Api)
    result = assets.prepare_environment(
        required_modules=(), assets=[{"kind": "model", "repo_id": "synthetic/original"}]
    )
    assert result["status"] == "BLOCKED" and result["code"] == "BLOCKED_ASSET"
    assert result["attempted"] and "403" in result["reason"]


def test_sae_download_scope_excludes_other_layers():
    assert assets._files(
        {"kind": "sae", "folder": "layer25/sae"},
        ["layer25/sae/params.npz", "layer13/sae/params.npz"],
    ) == ["layer25/sae/params.npz"]
    assert assets._files(
        {"kind": "sae", "folder": "layer20.sae.pt"}, ["layer20.sae.pt", "layer24.sae.pt"]
    ) == ["layer20.sae.pt"]


def config(tmp_path):
    prompt = tmp_path / "selected.json"
    prompt.write_text(json.dumps([{"prompt": "Synthetic selected dataset input"}]))
    return OmegaConf.create(
        {
            "model": {
                "name": "gemma2_2b_base",
                "hf_id": "google/gemma-2-2b",
                "default_saes": ["gemma2_2b_pt_l25_canonical"],
                "default_layers": {"gemma2_2b_pt_l25_canonical": 25},
            },
            "dataset": {"name": "IOI", "data_path": str(tmp_path), "prompt_file": str(prompt)},
            "paths": {"data_dir": str(tmp_path)},
            "seed": 42,
            "task": {
                "name": "precheck",
                "execute": True,
                "mode": "task",
                "check_tasks": ["capture"],
                "auto_setup": True,
                "prompt": "",
                "device": "cpu",
            },
        }
    )


def test_preparation_then_check_runs_without_secondary_confirmation(monkeypatch, tmp_path):
    from tasks.precheck import operation

    cfg = config(tmp_path)
    monkeypatch.chdir(tmp_path)
    order = []

    def worker(request, **kwargs):
        order.append(request["kind"])
        return {"status": "PASS", "scope": "synthetic worker fixture"}

    monkeypatch.setattr(operation, "_run_worker", worker)
    result = operation.run_precheck(cfg)
    assert result.ok and order == ["prepare_assets", "prepare_dataset", "dataset", "task"]
    assert result.meta["preparation"] and result.exit_code == 0


def test_hydra_precheck_builds_the_target_task_schema(tmp_path):
    from hydra import compose, initialize_config_module
    from tasks.precheck.operation import _task_config

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="config", overrides=["task=precheck", f"paths.root={tmp_path}"])
    assert OmegaConf.is_struct(cfg)
    task_cfg = _task_config(cfg, "capture", 300, 128)
    assert task_cfg["task"]["sae_key"] == "gemma2_2b_pt_l25_canonical"
    assert task_cfg["task"]["layer"] == 25
    assert task_cfg["task"]["top_feature_num"] == 10
    assert "auto_setup" not in task_cfg["task"]
    assert cfg.task.name == "precheck" and "sae_key" not in cfg.task


def test_failed_preparation_stops_real_checks_and_preserves_report(monkeypatch, tmp_path):
    from tasks.precheck import operation

    cfg = config(tmp_path)
    monkeypatch.chdir(tmp_path)
    order = []

    def worker(request, **kwargs):
        order.append(request["kind"])
        return {
            "status": "BLOCKED",
            "code": "BLOCKED_ASSET",
            "reason": "synthetic attempted failure",
        }

    monkeypatch.setattr(operation, "_run_worker", worker)
    result = operation.run_precheck(cfg)
    assert order == ["prepare_assets"] and result.exit_code == 2
    assert json.loads((tmp_path / "metadata/precheck_result.json").read_text())["preparation"]


@pytest.mark.parametrize("custom_root", [False, True])
def test_dataset_preparation_uses_owner_and_commits_complete_artifacts(
    monkeypatch, tmp_path, custom_root
):
    from saebase.datasets.prompts import prepare_selected_dataset
    from tests.product.test_reference_model_coverage import FakePythia
    from saebase import registry

    selected_root = tmp_path / "custom-inputs" if custom_root else tmp_path
    cfg = OmegaConf.create(
        {
            "model": {"name": "gemma2_2b_base", "device": "cpu"},
            "dataset": {
                "name": "SimpleCopying",
                "data_path": str(selected_root / "SimpleCopying"),
            },
            "paths": {"data_dir": str(tmp_path)},
            "seed": 0,
        }
    )
    calls = []

    def construct(config):
        calls.append("loaded")
        return SimpleNamespace(model=FakePythia(), loaded_name="synthetic-preparation-fixture")

    monkeypatch.setattr(
        registry, "get_model_plugin", lambda key: SimpleNamespace(load_model=construct)
    )
    result = prepare_selected_dataset(cfg, n_generate=4, min_kept=1)
    assert result["source"] == "prepared" and result["records"] == 4
    assert Path(result["path"]) == selected_root / "SimpleCopying/gemma2_2b_base/v1/prompts.jsonl"
    manifest = Path(result["path"]).parent / "manifest.json"
    assert json.loads(manifest.read_text())["filter_policy"]["name"] == "logit_diff"
    before = manifest.read_bytes()
    assert prepare_selected_dataset(cfg, n_generate=4, min_kept=1)["source"] == "existing"
    assert calls == ["loaded"] and manifest.read_bytes() == before


def test_dataset_preparation_never_replaces_partial_existing_artifacts(tmp_path):
    from saebase.datasets.prompts import prepare_selected_dataset

    cfg = OmegaConf.create(
        {
            "model": {"name": "gemma2_2b_base"},
            "dataset": {"name": "IOI", "data_path": str(tmp_path / "IOI")},
            "paths": {"data_dir": str(tmp_path)},
            "seed": 0,
        }
    )
    destination = tmp_path / "IOI/gemma2_2b_base/v1"
    destination.mkdir(parents=True)
    marker = destination / "previous.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare_selected_dataset(cfg, n_generate=4, min_kept=1)
    assert marker.read_text() == "preserve"


def test_registered_model_loading_is_distinct_from_lazy_metadata(monkeypatch):
    from models import hooked_lm_common
    from saebase.registry import get_model_plugin
    from saebase.api.sae_loading import load_model_for_profile

    sentinel = object()
    calls = []

    def load(spec, **kwargs):
        calls.append(spec.model_name)
        return SimpleNamespace(model=sentinel, loaded_name=spec.model_name)

    monkeypatch.setattr(hooked_lm_common, "load_from_spec", load)
    cfg = OmegaConf.create({"model": {"name": "gemma2_2b_base", "device": "cpu"}})
    plugin = get_model_plugin(cfg.model.name)
    assert plugin.construct(cfg)["lazy"] is True and not calls
    assert plugin.load_model(cfg).model is sentinel
    model, name = load_model_for_profile("gemma2_2b_pt_l25_canonical", device="cpu")
    assert model is sentinel and name == "google/gemma-2-2b"
    assert calls == [name, name]


def test_preparation_timeout_stops_the_worker_process_group(monkeypatch, tmp_path):
    from tasks.precheck import operation
    import signal

    class Process:
        pid = 12345

        def communicate(self, timeout=None):
            if timeout == 1:
                raise subprocess.TimeoutExpired("synthetic worker", timeout)
            return "synthetic partial output", "synthetic stderr"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(operation.subprocess, "Popen", lambda *args, **kwargs: Process())
    signals = []
    monkeypatch.setattr(operation.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    result = operation._run_worker(
        {"kind": "prepare_assets"}, label="setup", timeout=1, allow_network=True
    )
    assert result["status"] == "BLOCKED" and result["code"] == "BLOCKED_RESOURCE"
    assert signals == [(12345, signal.SIGTERM)]
    assert (
        tmp_path / "tmp/precheck_jobs/setup/logs/stdout.log"
    ).read_text() == "synthetic partial output"
