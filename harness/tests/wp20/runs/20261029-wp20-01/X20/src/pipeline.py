from __future__ import annotations

from pathlib import Path

from src.schema import require_keys


def execute(payload, workdir: Path):
    require_keys(payload, {"initial", "events"})
    state = dict(payload["initial"])
    if set(state) != {"revision", "source_sha", "owner", "status"}:
        raise ValueError("initial")
    history = []
    failures = []
    for event in payload["events"]:
        before = dict(state)
        kind = event["op"]
        if kind == "handoff":
            if event["from"] != state["owner"] or not event["to"]:
                raise ValueError("handoff")
            state["owner"] = event["to"]
            state["revision"] += 1
        elif kind == "change":
            if event["actor"] != "trusted-human" or event["expected"] != state["revision"]:
                raise ValueError("unauthorized change")
            state["source_sha"] = event["source_sha"]
            state["revision"] += 1
        elif kind == "fail":
            failures.append({"reason": event["reason"], "revision": state["revision"]})
            state["status"] = "partial"
        elif kind == "resume":
            if event["revision"] != state["revision"] or event["owner"] != state["owner"]:
                raise ValueError("stale recovery")
            state["status"] = "active"
        elif kind == "review":
            if event["actor"] != "trusted-human" or event["source_sha"] != state["source_sha"]:
                raise ValueError("approval")
            state["status"] = "reviewed"
        else:
            raise ValueError("event")
        history.append({"op": kind, "before": before, "after": dict(state)})
    return {
        "final": state,
        "events": history,
        "failures": failures,
        "human_approved": state["status"] == "reviewed",
        "count": len(history),
    }
