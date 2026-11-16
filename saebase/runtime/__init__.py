"""Import-light dependency queries and an explicit environment inspection command."""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from typing import Literal

Mode = Literal["core", "analysis", "generation"]

_PACKAGES: dict[Mode, tuple[tuple[str, str], ...]] = {
    "core": (
        ("hydra", "hydra-core"),
        ("omegaconf", "omegaconf"),
        ("torch", "torch"),
        ("numpy", "numpy"),
    ),
    "analysis": (
        ("transformer_lens", "transformer-lens"),
        ("sae_lens", "sae-lens"),
        ("huggingface_hub", "huggingface-hub"),
        ("safetensors", "safetensors"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
    ),
    "generation": (("vllm", "vllm==0.27.1"), ("nnsight", "nnsight[vllm]==0.8.0rc1")),
}


def required_packages(mode: Mode = "core") -> tuple[tuple[str, str], ...]:
    if mode not in _PACKAGES:
        raise ValueError(f"Unknown runtime mode: {mode!r}")
    if mode == "core":
        return _PACKAGES["core"]
    if mode == "analysis":
        return _PACKAGES["core"] + _PACKAGES["analysis"]
    return _PACKAGES["core"] + _PACKAGES["analysis"] + _PACKAGES["generation"]


def missing_packages(mode: Mode = "core") -> list[tuple[str, str]]:
    """Probe presence without importing heavyweight models or GPU backends."""
    return [(mod, pip) for mod, pip in required_packages(mode) if find_spec(mod) is None]


def print_python_identity() -> None:
    print("[env] python:", sys.executable)
    print("[env] version:", sys.version.split()[0])


def verify_runtime(*, context: str = "SAEBase", mode: Mode = "core") -> None:
    missing = missing_packages(mode)
    if not missing:
        return
    requirement = {"core": ".", "analysis": ".[analysis]", "generation": ".[generation]"}[mode]
    details = ", ".join(f"{mod} ({pip})" for mod, pip in missing)
    raise RuntimeError(
        f"{context}: missing dependencies for {mode}: {details}. "
        f'Run `{sys.executable} -m pip install -e "{requirement}"` from the repository in the interpreter '
        "that runs SAEBase."
    )


def __getattr__(name: str):
    if name == "TaskModule":
        from saebase.registry import TaskModule

        return TaskModule
    if name == "apply_lightning_reproducibility":
        from saebase.runtime.training.seed import apply_lightning_reproducibility

        return apply_lightning_reproducibility
    if name == "verify_training_runtime":
        from saebase.runtime.runtime_check import verify_training_runtime

        return verify_training_runtime
    raise AttributeError(name)


def main():
    p = argparse.ArgumentParser(description="Inspect SAEBase dependencies")
    p.add_argument("--mode", choices=("core", "analysis", "generation"), default="analysis")
    a = p.parse_args()
    print_python_identity()
    missing = missing_packages(a.mode)
    for mod, package in required_packages(a.mode):
        print(f"[{'missing' if any(x[0] == mod for x in missing) else 'ok'}] {mod} ({package})")
    if missing:
        extra = {"core": ".", "analysis": ".[analysis]", "generation": ".[generation]"}[a.mode]
        print(f'[env] Install from the repository: python -m pip install -e "{extra}"')
        return 1
    print(
        f"[env] {a.mode} dependency probes passed (checkpoint download and GPU runtime not tested)"
    )
    return 0
