"""Import-light dependency checks scoped to the requested SAEBase capability."""
from __future__ import annotations

from importlib.util import find_spec
import sys
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
    requirement = {
        "core": ".", "analysis": ".[analysis]", "generation": ".[generation]"
    }[mode]
    details = ", ".join(f"{mod} ({pip})" for mod, pip in missing)
    raise RuntimeError(
        f"{context}: missing dependencies for {mode}: {details}. "
        f"Run `{sys.executable} -m pip install -e \"{requirement}\"` from the repository in the interpreter "
        "that runs SAEBase."
    )
