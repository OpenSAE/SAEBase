"""Apply one local draft decision. The result is never a formal approval."""

from __future__ import annotations

_STATES = {"unchecked", "approve", "deny"}


class InputError(ValueError):
    """Raised when a revision is stale or a decision would grant formal authority."""


def solve(payload: dict) -> dict:
    """Update one node draft and increment revision. approved stays false."""
    if not isinstance(payload, dict) or set(payload) != {"revision", "drafts", "event"}:
        raise InputError("schema")
    revision = payload["revision"]
    drafts = payload["drafts"]
    event = payload["event"]
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise InputError("revision")
    if not isinstance(drafts, dict) or not drafts:
        raise InputError("drafts")
    decision, node_id, event_revision = _event(event)
    if event_revision != revision or node_id not in drafts:
        raise InputError("revision")
    current: dict[str, str] = {}
    for key, state in drafts.items():
        if not isinstance(key, str) or not key or state not in _STATES:
            raise InputError("drafts")
        current[key] = state
    current[node_id] = decision
    return {"revision": revision + 1, "drafts": current, "approved": False}


def _event(event: object) -> tuple[str, str, int]:
    """Read a node-local decision. formal_approve is rejected."""
    if not isinstance(event, dict) or set(event) != {"id", "decision", "revision"}:
        raise InputError("event")
    node_id = event["id"]
    decision = event["decision"]
    revision = event["revision"]
    if not isinstance(node_id, str) or not node_id:
        raise InputError("id")
    if decision not in _STATES:
        raise InputError("decision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise InputError("revision")
    return decision, node_id, revision
