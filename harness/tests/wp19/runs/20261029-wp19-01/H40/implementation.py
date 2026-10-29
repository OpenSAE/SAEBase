"""Compare two source SHAs. A mismatch is stale, not a silent reuse."""

from __future__ import annotations

import re

SHA = re.compile(r"[0-9a-f]{40,64}$")


class InputError(ValueError):
    """Raised for a malformed digest or a checkout that does not match the expected SHA."""


def solve(payload: dict) -> dict:
    """Return matched true only when both lowercase hex digests are identical."""
    if not isinstance(payload, dict) or set(payload) != {"expected_sha", "observed_sha"}:
        raise InputError("schema")
    expected = _sha(payload["expected_sha"])
    observed = _sha(payload["observed_sha"])
    if expected != observed:
        raise InputError("stale")
    return {"matched": True}


def _sha(value: object) -> str:
    """Accept one 40-to-64 character lowercase hexadecimal digest."""
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise InputError("sha")
    return value
