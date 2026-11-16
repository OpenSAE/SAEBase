"""GPU queue contract checks without CUDA, weights, tmux or real waiting."""

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sar_gpu_queue", ROOT / "scripts/capture_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def settings(**values):
    return Namespace(
        gpu="0", max_used_mib=1024, max_utilization=5, idle_seconds=30, poll_seconds=10, **values
    )


def snapshot(*, pids=(), memory=0, utilization=0):
    return {
        "uuid": "GPU-test",
        "compute_pids": list(pids),
        "used_mib": memory,
        "utilization": utilization,
    }


@pytest.mark.parametrize(
    "value", [snapshot(pids=[1]), snapshot(memory=2048), snapshot(utilization=40)]
)
def test_foreign_activity_is_not_idle(value):
    assert not queue.is_idle(value, 1024, 5)


def test_wait_requires_continuous_quiet_and_never_kills_foreign_process(tmp_path, monkeypatch):
    elapsed = [0]
    samples = iter(
        [
            snapshot(pids=[99]),
            snapshot(),
            snapshot(),
            snapshot(utilization=80),
            snapshot(),
            snapshot(),
            snapshot(),
            snapshot(),
        ]
    )
    monkeypatch.setattr(
        queue.os, "killpg", lambda *args: pytest.fail("must not kill foreign processes")
    )
    result = queue.wait_for_gpu(
        settings(),
        tmp_path / "state.json",
        {},
        sample=lambda _: next(samples),
        sleep=lambda _: elapsed.__setitem__(0, elapsed[0] + 10),
        now=lambda: elapsed[0],
    )
    assert result["uuid"] == "GPU-test" and elapsed[0] == 70
    assert json.loads((tmp_path / "state.json").read_text())["quiet_seconds"] == 30


def test_pid_birth_identity_blocks_gap_in_previous_batch(tmp_path, monkeypatch):
    elapsed = [0]
    monkeypatch.setattr(
        queue, "process_identity", lambda pid: "original" if elapsed[0] < 20 else "new_pid_owner"
    )
    queue.wait_for_gpu(
        settings(),
        tmp_path / "state.json",
        {55: "original"},
        sample=lambda _: snapshot(),
        sleep=lambda _: elapsed.__setitem__(0, elapsed[0] + 10),
        now=lambda: elapsed[0],
    )
    assert elapsed[0] == 50


def test_failed_nvidia_probe_does_not_launch(tmp_path):
    elapsed = [0]

    def sample(_):
        if elapsed[0] == 0:
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return snapshot()

    queue.wait_for_gpu(
        settings(),
        tmp_path / "state.json",
        {},
        sample=sample,
        sleep=lambda _: elapsed.__setitem__(0, elapsed[0] + 10),
        now=lambda: elapsed[0],
    )
    assert elapsed[0] == 40


def test_sar_matrix_disables_prefix_only_offload():
    from tasks.capture.matrix import CaptureMatrix

    matrix = CaptureMatrix(ROOT / "conf/experiment/capture_matrix.yaml", root=ROOT, mode="sar")
    assert len(matrix.MODELS) == 10 and len(matrix.DATASETS) == 12
    assert not any(matrix.MEMORY.values())
    with pytest.raises(ValueError, match="complete downstream"):
        matrix.child(matrix.MODELS[0][0], offload_unused_layers=True)


def test_explicit_model_revision_composes_with_hydra_struct_mode():
    from tasks.capture.matrix import CaptureMatrix
    from hydra import initialize_config_module

    matrix = CaptureMatrix(ROOT / "conf/experiment/capture_matrix.yaml", root=ROOT, mode="sar")
    matrix.MODEL_OVERRIDES["gpt2small"] = {"revision": "6" * 40}
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = matrix.capture_config("hotpotqa", "gpt2small", "gpt2small_res_jb_l8", 8)
        assert cfg.model.revision == "6" * 40
        another = matrix.capture_config("hotpotqa", "llama31_8b", "llama31_8b_lxr_8x", 16)
        assert "revision" not in another.model


@pytest.mark.parametrize("seconds", [0, 60, 299])
def test_cli_rejects_idle_threshold_below_five_minutes(tmp_path, capsys, seconds):
    with pytest.raises(SystemExit) as caught:
        queue.main(
            [
                "--sar-source",
                str(tmp_path / "sar.py"),
                "--report-dir",
                str(tmp_path / "report"),
                "--tag",
                "test",
                "--idle-seconds",
                str(seconds),
            ]
        )
    assert caught.value.code == 2
    assert "at least 300 seconds" in capsys.readouterr().err


def test_default_idle_threshold_is_five_minutes(tmp_path, monkeypatch):
    class ConfigReached(Exception):
        pass

    def capture_configuration(args, report):
        assert args.idle_seconds == 300
        raise ConfigReached

    monkeypatch.setattr(queue, "freeze_config", capture_configuration)
    with pytest.raises(ConfigReached):
        queue.main(
            [
                "--sar-source",
                str(tmp_path / "sar.py"),
                "--report-dir",
                str(tmp_path / "report"),
                "--tag",
                "test",
            ]
        )
