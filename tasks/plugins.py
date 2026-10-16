"""Compatibility shim for historical tasks.plugins imports."""

from __future__ import annotations

from saebase.tasks import plugins as _canonical


def __getattr__(name: str) -> object:
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_canonical)))
