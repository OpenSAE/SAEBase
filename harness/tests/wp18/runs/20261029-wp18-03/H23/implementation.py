"""Decide whether one synthetic release has a matching SHA, successful checks, and approval."""

from __future__ import annotations

import re

_SHA = re.compile(r"[0-9a-f]{64}")
_REQUIRED = ("quality", "tests")


class InputError(ValueError):
    """Raised when a release lacks a matching SHA, a successful required check, or approval."""


def solve(payload: dict) -> dict:
    """Publish only when every supplied check succeeded and a human already approved it."""
    expected = {"tested_sha", "release_sha", "checks", "approved"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise InputError("schema")
    tested = _sha(payload["tested_sha"])
    release = _sha(payload["release_sha"])
    if tested != release:
        raise InputError("sha")
    checks = payload["checks"]
    if not isinstance(checks, dict) or not checks:
        raise InputError("checks")
    for name in _REQUIRED:
        if name not in checks:
            raise InputError("checks")
    for name, status in checks.items():
        if not isinstance(name, str) or not name or status != "success":
            raise InputError("checks")
    if payload["approved"] is not True:
        raise InputError("approval")
    return {"publish": True, "sha": release}


def _sha(value: object) -> str:
    """Accept one lowercase 64-character source digest."""
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise InputError("sha")
    return value
