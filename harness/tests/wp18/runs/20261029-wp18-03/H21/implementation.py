"""Accept direct SAEBase module names and reject retired domain layers."""

from __future__ import annotations

import re

_ROOTS = {"conf", "datasets", "harness", "models", "saebase", "tasks"}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class InputError(ValueError):
    """Raised for a retired root, an intermediate domain, or a duplicate module."""


def solve(payload: dict) -> dict:
    """Return unique module names in dictionary order."""
    if not isinstance(payload, dict) or set(payload) != {"modules"}:
        raise InputError("schema")
    modules = payload["modules"]
    if not isinstance(modules, list):
        raise InputError("modules")
    accepted: list[str] = []
    seen: set[str] = set()
    for name in modules:
        clean = _module_name(name)
        if clean in seen:
            raise InputError("duplicate")
        seen.add(clean)
        accepted.append(clean)
    return {"modules": sorted(accepted)}


def _module_name(value: object) -> str:
    """Allow exactly one owned root plus one module segment."""
    if not isinstance(value, str):
        raise InputError("module")
    parts = value.split(".")
    if len(parts) != 2 or parts[0] not in _ROOTS:
        raise InputError("module")
    if any(_NAME.fullmatch(part) is None for part in parts):
        raise InputError("module")
    return value
