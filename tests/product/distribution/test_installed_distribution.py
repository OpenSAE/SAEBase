"""Build and install a development wheel offline, then exercise it outside the checkout."""

import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_wheel_and_sdist_preserve_offline_task_interfaces(tmp_path):
    # setuptools mutates sys.argv during builds. Remove only the exact build
    # artifact left by the earlier two-backend-call regression, never source.
    legacy = ROOT / "bdist_wheel"
    if legacy.is_dir() and not legacy.is_symlink():
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        artifact = legacy / f"opensae_saebase-{version}.tar.gz"
        if list(legacy.iterdir()) == [artifact] and not artifact.is_symlink():
            artifact.unlink()
            legacy.rmdir()
    environment = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    # Reuse a dirty build output, as happens after deleting a source module.
    stale = ROOT / ".build/package/lib/tasks/generate/vllm_backend/generate.py"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("RETIRED_BUILD_SENTINEL = True\n")
    dist = tmp_path / "dist"
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel, build_sdist; import sys; "
            "destination = sys.argv[1]; build_wheel(destination); build_sdist(destination)",
            str(dist),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    assert not stale.exists(), "The wheel build must remove retired module output"
    assert not legacy.exists(), (
        "Backend output must stay inside the temporary distribution directory"
    )
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.startswith("doc/") or "vllm_backend/" in name for name in archive.namelist())
        assert "tasks/generate/serving/decoder_steering.py" in archive.namelist()
        masters = {
            name
            for name in archive.namelist()
            if name.startswith("conf/") and name.count("/") == 1 and name.endswith(".yaml")
        }
    assert masters == {"conf/config.yaml"}
    from scripts.distribution import validate

    validate(dist)
    source = next(dist.glob("*.tar.gz"))
    with tarfile.open(source) as archive:
        members = archive.getnames()
    assert any(name.endswith("tasks/capture/operation.py") for name in members)
    assert any(name.endswith("base/runtime/experiment_results.py") for name in members)
    assert any(name.endswith("conf/config.yaml") for name in members)
    assert not any(
        name.endswith(("conf/benchmark.yaml", "conf/sae_research.yaml")) for name in members
    )
    site = tmp_path / "installed"
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(site),
            str(wheel),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert installed.returncode == 0, installed.stderr
    import json

    from tests.product.benchmarks.test_benchmark_launch import protocol

    (tmp_path / "protocol.json").write_text(json.dumps(protocol()))
    script = """
import importlib.metadata
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import saebase
import saebase.datasets
import tasks.capture.matrix
import tasks.precheck.checkpoint
import tasks.intervene.commands
import tasks.benchmark.commands
import tasks.capture.response_surface
import tasks.capture.sar_mode
from saebase.api.paper_sources import load_paper_module
from tasks.benchmark.methods import evaluate_sar
assert evaluate_sar is tasks.capture.response_surface.evaluate_sar
assert Path(tasks.capture.sar_mode.__file__).is_relative_to(Path(sys.argv[1]))
from hydra import initialize_config_module, compose
from saebase.runner import run_task
from saebase.runtime.configuration.config_validate import validate_hydra_cfg
from saebase.results import EvaluationResult
assert Path(saebase.__file__).is_relative_to(Path(sys.argv[1]))
assert Path(saebase.datasets.__file__).is_relative_to(Path(sys.argv[1]))
with initialize_config_module(version_base=None, config_module="conf"):
    cfg = compose(config_name="config")
validate_hydra_cfg(cfg)
result = run_task(cfg)
assert isinstance(result, EvaluationResult)
assert result.status == "configured_only" and result.exit_code == 0
import saebase.runtime.train
from saebase.runtime.experiment_paths import run_leaf
with initialize_config_module(version_base=None, config_module="conf"):
    benchmark = compose(config_name="config", overrides=[
        "task=benchmark", "model=pythia70m",
        f"task.benchmark.benchmark_config={Path.cwd() / 'protocol.json'}", f"paths.root={Path.cwd()}"
    ])
validate_hydra_cfg(benchmark)
assert benchmark.model.name == "pythia70m" and benchmark.dataset.name == "IOI"
assert run_task(benchmark).model_key == "pythia70m"
assert run_leaf(benchmark).parts[-4:] == ("experiment", "IOI", "pythia70m", "benchmark")
entrypoints = importlib.metadata.distribution("opensae-saebase").entry_points
assert any(ep.name == "saebase-benchmark" for ep in entrypoints)
from saebase import cli
sys.argv = ["saebase", "--cfg", "job", "task=intervene"]
cli.main()
"""
    checked = subprocess.run(
        [sys.executable, "-I", "-c", script, str(site)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "name: intervene" in checked.stdout
