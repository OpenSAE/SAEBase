"""Stable CLI entrypoints under the saebase namespace.

Historical main.py and scripts.* modules remain importable. Import the
command implementation only when the corresponding CLI is invoked, so
ordinary saebase imports do not require optional analysis backends.
"""

from __future__ import annotations

from importlib import import_module


def _dispatch(module: str) -> object:
    return import_module(module).main()


def main() -> object:
    return _dispatch("main")


def prepare_prompts() -> object:
    return _dispatch("scripts.sae_prepare_prompts")


def check_env() -> object:
    return _dispatch("scripts.check_env")


def intervene() -> object:
    return _dispatch("scripts.m2_batch")


def benchmark() -> object:
    return _dispatch("scripts.m3_benchmark")


def audit() -> object:
    return _dispatch("scripts.m4_reliability")


def ground_truth() -> object:
    return _dispatch("scripts.m4_ground_truth")


def release_matrix() -> object:
    return _dispatch("scripts.m5_release_matrix")


def release_gate() -> object:
    return _dispatch("scripts.m5_release_gate")
