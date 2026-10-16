"""Compatibility shim for historical models.plugins imports.

New code should import saebase.model_plugins. This module delegates all
attributes to that single registration implementation.
"""

from __future__ import annotations

from saebase import model_plugins as _canonical


def __getattr__(name: str) -> object:
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_canonical)))
