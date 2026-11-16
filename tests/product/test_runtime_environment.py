"""Real subprocess checks for model environment selection and owned-process cleanup."""

import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest
import yaml

from saebase.runtime.environment import redirect_model, run_model


@pytest.fixture
def environments(tmp_path):
    alternate = tmp_path / "alternate"
    venv.EnvBuilder(system_site_packages=True).create(alternate)
    config = tmp_path / "environments.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "default": {"python": sys.executable},
                "models": {"alternate": {"python": str(alternate / "bin/python")}},
            }
        )
    )
    return config, alternate


def test_default_alternate_default_keep_parent_and_package_search_intact(
    environments, tmp_path, monkeypatch
):
    config, alternate = environments
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "foreign-packages"))
    before = dict(os.environ)
    prefixes = []
    for index, model in enumerate(["ordinary", "alternate", "ordinary"]):
        log = tmp_path / f"{index}.log"
        arguments = [
            "-c",
            "import sys,os,json; print(json.dumps([sys.prefix,os.getenv('PYTHONPATH')]))",
        ]
        if model == "alternate":
            assert redirect_model(model, arguments, log, cwd=tmp_path, config=config) == 0
        else:
            result = run_model(model, arguments, log, cwd=tmp_path, config=config)
            assert result["exit_code"] == 0
        prefix, inherited = json.loads(log.read_text())
        assert inherited is None
        prefixes.append(prefix)
    assert prefixes == [sys.prefix, str(alternate), sys.prefix]
    assert dict(os.environ) == before


@pytest.mark.parametrize("direct", [False, True])
def test_version_mismatch_prevents_payload_execution(environments, tmp_path, direct):
    config, _ = environments
    cfg = yaml.safe_load(config.read_text())
    cfg["default"]["packages"] = {"transformers": "0.invalid"}
    config.write_text(yaml.safe_dump(cfg))
    log = tmp_path / "never-started.log"
    launch = redirect_model if direct else run_model
    launch("ordinary", ["-c", "raise AssertionError('started')"], log, cwd=tmp_path, config=config)
    result = json.loads(log.with_suffix(".environment.json").read_text())
    assert result["status"] == "ENVIRONMENT_ERROR"
    assert "expected 0.invalid" in result["error"]
    assert not log.exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_owned_descendants_are_stopped_on_timeout_or_parent_interrupt(
    environments, tmp_path, interrupt
):
    config, _ = environments
    pidfile = tmp_path / "descendant.pid"
    # This descendant ignores SIGTERM; group cleanup must also remove it.
    descendant = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    child = (
        "import subprocess,sys,os,signal,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
        f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(.1); "
        + ("os.kill(os.getppid(),signal.SIGTERM); " if interrupt else "")
        + "time.sleep(60)"
    )
    wrapper = (
        "from saebase.runtime.environment import run_model\n"
        "try:\n"
        f" run_model('ordinary', ['-c', {child!r}], {str(tmp_path / 'child.log')!r}, "
        f"cwd={str(tmp_path)!r}, config={str(config)!r}, timeout=.5)\n"
        "except KeyboardInterrupt: pass\n"
    )
    subprocess.run([sys.executable, "-c", wrapper], check=True, timeout=30)
    pid = int(pidfile.read_text())
    proc = Path(f"/proc/{pid}/stat")
    assert not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    receipt = json.loads((tmp_path / "child.environment.json").read_text())
    assert receipt["status"] == ("INTERRUPTED" if interrupt else "TIMEOUT")


def test_nonzero_child_exit_preserves_failure_and_next_default_run(environments, tmp_path):
    config, _ = environments
    failed = run_model(
        "alternate",
        ["-c", "raise SystemExit(7)"],
        tmp_path / "bad.log",
        cwd=tmp_path,
        config=config,
    )
    following = run_model(
        "ordinary", ["-c", "pass"], tmp_path / "good.log", cwd=tmp_path, config=config
    )
    assert failed["exit_code"] == 7 and following["exit_code"] == 0
    assert following["environment"]["prefix"] == sys.prefix
