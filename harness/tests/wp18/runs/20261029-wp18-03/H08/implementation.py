"""Mark claims resolved only when the caller sets observed to true."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a claim is malformed, duplicated, or points outside the repo."""


def solve(payload: dict) -> dict:
    """Split claim ids into resolved and unresolved lists, preserving input order."""
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        raise InputError("schema")
    claims = payload["claims"]
    if not isinstance(claims, list):
        raise InputError("claims")
    resolved: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        claim_id, observed = _claim(claim)
        if claim_id in seen:
            raise InputError("duplicate")
        seen.add(claim_id)
        if observed:
            resolved.append(claim_id)
        else:
            unresolved.append(claim_id)
    return {"resolved": resolved, "unresolved": unresolved}


def _claim(claim: object) -> tuple[str, bool]:
    """Accept one id, one repo-relative ref, and a real boolean observed flag."""
    if not isinstance(claim, dict) or set(claim) != {"id", "ref", "observed"}:
        raise InputError("claim")
    claim_id = claim["id"]
    ref = claim["ref"]
    observed = claim["observed"]
    if not isinstance(claim_id, str) or not claim_id or claim_id in {".", ".."}:
        raise InputError("id")
    if not isinstance(observed, bool):
        raise InputError("observed")
    if not isinstance(ref, str) or ref.count(":") != 1 or "\\" in ref or ref.startswith("/"):
        raise InputError("ref")
    path, line = ref.split(":", 1)
    if not line.isdigit() or int(line) < 1:
        raise InputError("line")
    parts = path.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InputError("ref")
    return claim_id, observed
