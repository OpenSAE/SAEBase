"""Current CLI entrypoints; optional scientific backends load only when selected."""

from __future__ import annotations

from importlib import import_module


def _dispatch(module: str) -> object:
    return import_module(module).main()


def main() -> object:
    return _dispatch("saebase.runtime.train")


def prepare_prompts() -> object:
    return _dispatch("saebase.datasets.prepare")


def check_env() -> object:
    return _dispatch("saebase.runtime")


def intervene() -> object:
    return _dispatch("tasks.intervene.commands")


def benchmark() -> object:
    return _dispatch("tasks.benchmark.commands")


def audit() -> object:
    return _dispatch("tasks.benchmark.reliability.commands")


def ground_truth() -> object:
    return _dispatch("tasks.benchmark.reliability.controls_cli")
