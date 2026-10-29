from __future__ import annotations

from pathlib import Path

from src.schema import require_keys

REQUIRED = {"core", "knowledge", "code", "environment", "context"}


def execute(payload, workdir: Path):
    require_keys(payload, {"trees", "events"})
    trees = {
        name: {"focus": None, "expanded": [], "decisions": {}, "revision": 0}
        for name in payload["trees"]
    }
    if set(trees) != REQUIRED:
        raise ValueError("tree set")
    active, language, history = "core", "en", []
    for event in payload["events"]:
        op, tree = event["op"], event.get("tree", active)
        if tree not in trees:
            raise ValueError("tree")
        if op == "switch":
            active = tree
        elif op == "language":
            if event["value"] not in ("zh", "en"):
                raise ValueError("language")
            language = event["value"]
        elif op in ("focus", "toggle"):
            if event["id"] not in payload["trees"][tree]:
                raise ValueError("unknown node")
            if op == "focus":
                trees[tree]["focus"] = event["id"]
            else:
                expanded = trees[tree]["expanded"]
                if event["id"] in expanded:
                    expanded.remove(event["id"])
                else:
                    expanded.append(event["id"])
                    expanded.sort()
        elif op == "review":
            state = trees[tree]
            if (
                event["id"] not in payload["trees"][tree]
                or event["revision"] != state["revision"]
                or event["decision"] not in ("approve", "deny", "unchecked")
            ):
                raise ValueError("review")
            state["decisions"][event["id"]] = event["decision"]
            state["revision"] += 1
        else:
            raise ValueError("operation")
        history.append(
            {
                "op": op,
                "active": active,
                "language": language,
                "revision": trees[tree]["revision"],
            }
        )
    return {
        "active": active,
        "language": language,
        "trees": trees,
        "event_count": len(history),
        "events": history,
    }
