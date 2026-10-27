"""CPU-first dependency checks for the SAE research configuration."""

from __future__ import annotations
import sys
from typing import List, Tuple

REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("torch", "torch"),
)


def missing_packages(*, profile: str = "llm") -> List[Tuple[str, str]]:
    return [(mod, pip) for mod, pip in REQUIRED if not _available(mod)]


def _available(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def print_python_identity() -> None:
    print("[env] python:", sys.executable)
    print("[env] version:", sys.version.split()[0])


def verify_training_runtime(*, context: str = "SAEBase", profile: str = "llm") -> None:
    print_python_identity()
    missing = missing_packages(profile=profile)
    if missing:
        raise SystemExit(f"Missing {context} dependencies: {missing}")
