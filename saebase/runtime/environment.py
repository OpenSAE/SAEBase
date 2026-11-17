"""Select checked Python environments and run owned experiment subprocesses.

Scope: model launchers share repository configuration, isolated child variables,
environment receipts and process-group cleanup. The parent environment stays intact.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


PACKAGES = ("torch", "transformers", "huggingface-hub", "tokenizers", "sae-lens", "threadpoolctl")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_THREADPOOL_LIMITER = None


def worker_environment(cpu_threads=4):
    """Bound independent CPU pools in research workers; explicit Runtime settings win."""
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("runtime.cpu_threads must be a positive integer")
    return {
        **dict.fromkeys(THREAD_VARIABLES, str(cpu_threads)),
        "TOKENIZERS_PARALLELISM": "false",
    }


def configure_worker(cfg):
    """Apply CPU settings before model loading, including already-loaded BLAS pools.

    Configuration-only calls and legacy configurations keep the host untouched.
    CUDA arithmetic belongs to models.numerics, independently of CPU scheduling.
    """
    if not cfg.get("task", {}).get("execute", False):
        return
    count = cfg.get("runtime", {}).get("cpu_threads")
    if count is None:
        return
    from threadpoolctl import threadpool_limits
    import torch

    global _THREADPOOL_LIMITER
    os.environ.update(worker_environment(count))
    torch.set_num_threads(count)
    _THREADPOOL_LIMITER = threadpool_limits(limits=count)


def worker_snapshot():
    """Save the effective CPU execution settings rather than only the request."""
    from threadpoolctl import threadpool_info
    import torch

    return {
        "torch_threads": torch.get_num_threads(),
        "threadpools": [
            {key: pool.get(key) for key in ("internal_api", "prefix", "num_threads")}
            for pool in threadpool_info()
        ],
        "environment": {
            key: os.environ.get(key) for key in (*THREAD_VARIABLES, "TOKENIZERS_PARALLELISM")
        },
    }


def snapshot_environment(architectures=()):
    """Record the interpreter and the actual locations of research dependencies."""
    packages = {}
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
        else:
            packages[name] = {"version": dist.version, "path": str(dist.locate_file(""))}
    supported = {}
    if architectures:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        for name in architectures:
            try:
                supported[name] = CONFIG_MAPPING[name].__name__
            except KeyError as exc:
                raise ValueError(f"Python {sys.executable} does not support {name}") from exc
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "packages": packages,
        "architectures": supported,
        "worker_environment": {
            key: os.environ.get(key) for key in (*THREAD_VARIABLES, "TOKENIZERS_PARALLELISM")
        },
    }


def child_environment(python, extra=None):
    """Build child-only variables, removing inherited Python package search overrides."""
    environment = {**os.environ, **worker_environment(), **(extra or {})}
    for key in ("PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX"):
        environment.pop(key, None)
    prefix = Path(python).parent.parent
    environment.update(
        VIRTUAL_ENV=str(prefix),
        PATH=str(Path(python).parent) + os.pathsep + environment.get("PATH", ""),
        PYTHONNOUSERSITE="1",
    )
    return environment


def select_environment(model, *, config=None):
    """Resolve a model's configured interpreter without dereferencing venv symlinks."""
    from importlib.resources import files
    import yaml

    path = Path(config) if config else Path(str(files("conf") / "runtime/environments.yaml"))
    settings = yaml.safe_load(path.read_text())
    profile = settings.get("models", {}).get(model, settings["default"])
    python = profile["python"]
    python = os.path.abspath(os.path.expanduser(str(python)))
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Configured Python for {model} is not executable: {python}")
    return python, profile


def prepare_launch(model, arguments, *, config=None, environment=None):
    """Check package versions and architecture support before any model weights load."""
    python, profile = select_environment(model, config=config)
    child_env = child_environment(python, environment)
    probe = subprocess.run(
        [
            python,
            "-I",
            "-c",
            "import json,runpy,sys; "
            "m=runpy.run_path(sys.argv[1]); "
            "print(json.dumps(m['snapshot_environment'](json.loads(sys.argv[2]))))",
            str(Path(__file__).absolute()),
            json.dumps(profile.get("architectures", [])),
        ],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode:
        raise RuntimeError(f"Environment check failed for {model}: {probe.stderr.strip()}")
    observed = json.loads(probe.stdout)
    for name, expected in profile.get("packages", {}).items():
        actual = (observed["packages"].get(name) or {}).get("version")
        if actual != str(expected):
            raise ValueError(
                f"Environment for {model}: {name} expected {expected}, got {actual} ({python})"
            )
    return [python, *map(str, arguments)], child_env, observed


def redirect_model(model, arguments, log, *, cwd, config=None):
    """Route a directly invoked model worker through its configured environment once."""
    try:
        python, _ = select_environment(model, config=config)
        if Path(sys.prefix).absolute() == Path(python).parent.parent:
            prepare_launch(model, [], config=config)
            return None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        receipt = Path(log).with_suffix(".environment.json")
        return _environment_failure({"model": model}, receipt, exc)["exit_code"]
    return run_model(model, arguments, log, cwd=cwd, config=config)["exit_code"]


def _environment_failure(outcome, receipt, error):
    from saebase.runtime.experiment_results import write_json

    outcome.update(status="ENVIRONMENT_ERROR", exit_code=125, error=str(error))
    write_json(receipt, outcome)
    return outcome


def stop_child(process):
    """Terminate this launch's process group, including descendants left by its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_model(
    model,
    arguments,
    log,
    *,
    cwd,
    timeout=None,
    config=None,
    environment=None,
    record_path=None,
):
    """Launch, record and clean up a model child on success, failure, timeout or interruption."""
    from saebase.runtime.experiment_results import write_json

    log = Path(log)
    receipt = Path(record_path) if record_path else log.with_suffix(".environment.json")
    outcome = {"model": model, "log": str(log), "environment_record": str(receipt)}
    try:
        command, child_env, observed = prepare_launch(
            model,
            arguments,
            config=config,
            environment=environment,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return _environment_failure(outcome, receipt, exc)
    outcome.update(command=command, environment=observed, status="RUNNING")
    write_json(receipt, outcome)
    log.parent.mkdir(parents=True, exist_ok=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, interrupt) if main_thread else None
    process = None
    try:
        with log.open("a") as output:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )
            outcome.update(status="FINISHED", exit_code=process.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        outcome.update(status="TIMEOUT", timeout_seconds=timeout, exit_code=124)
    except KeyboardInterrupt:
        outcome.update(status="INTERRUPTED", exit_code=130)
        raise
    except OSError as exc:
        outcome.update(status="LAUNCH_ERROR", exit_code=125, error=str(exc))
    finally:
        if process is not None:
            stop_child(process)
        if main_thread:
            signal.signal(signal.SIGTERM, previous)
        write_json(receipt, outcome)
    return outcome
