"""Small governance-owned CPU reference for WP19 H27-H46.

Reference behavior is public; independent reviewers must still test different
inputs. Never import this from task-agent code and claim their result is verified.
"""

from __future__ import annotations

import re

from harness.contracts import registry
from harness.contracts.selection import SIGNALS

ALWAYS = tuple(registry.catalog()["required_always"])
KNOWN = {row["id"] for row in registry.catalog()["contracts"]}
SHA = re.compile(r"[0-9a-f]{40,64}$")


def req(payload: object, names: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != names:
        raise ValueError("schema")
    return payload


def words(value: object, *, known: set[str] | None = None) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(s, str) or not s for s in value):
        raise ValueError("list")
    if len(set(value)) != len(value):
        raise ValueError("duplicate")
    if known is not None and not set(value) <= known:
        raise ValueError("unknown")
    return value


def solve(case: str, payload: dict) -> dict:
    if case == "H27":
        p = req(payload, {"selected"})
        return {"contracts": sorted(set(ALWAYS) | set(words(p["selected"], known=KNOWN)))}
    if case == "H28":
        p = req(payload, {"global_min", "local_min"})
        a, b = p["global_min"], p["local_min"]
        if type(a) is not int or type(b) is not int or a < 0 or b < a:
            raise ValueError("weaken")
        return {"minimum": b}
    if case == "H29":
        p = req(payload, {"ids"})
        return {"ids": sorted(words(p["ids"], known=KNOWN))}
    if case == "H30":
        p = req(payload, {"expected", "observed"})
        if not all(isinstance(p[k], str) and p[k] for k in p) or p["expected"] != p["observed"]:
            raise ValueError("version")
        return {"matched": True, "version": p["expected"]}
    if case == "H31":
        p = req(payload, {"selected", "signals"})
        selected = words(p["selected"], known=KNOWN)
        signals = words(p["signals"], known=set(SIGNALS))
        return {"missing": sorted({SIGNALS[x] for x in signals} - set(selected))}
    if case == "H32":
        p = req(payload, {"groups"})
        groups = p["groups"]
        if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
            raise ValueError("groups")
        merged = set(ALWAYS)
        for group in groups:
            merged.update(words(group, known=KNOWN))
        return {"contracts": sorted(merged)}
    if case == "H33":
        p = req(payload, {"binding"})
        if p["binding"] is None:
            return {"status": "legacy_unbound", "verified": False}
        words(p["binding"], known=KNOWN)
        if not p["binding"]:
            raise ValueError("empty")
        return {"status": "bound", "verified": True}
    if case == "H34":
        p = req(payload, {"action", "authorized"})
        if p["action"] not in ("sandbox_write", "governance_write") or type(p["authorized"]) is not bool:
            raise ValueError("authorization")
        return {"allowed": p["action"] == "sandbox_write"}
    if case == "H35":
        p = req(payload, {"rules"})
        rows = p["rules"]
        if not isinstance(rows, list):
            raise ValueError("rules")
        by_path: dict[str, set[bool]] = {}
        for r in rows:
            if not isinstance(r, dict) or set(r) != {"path", "allow"} or not isinstance(r["path"], str) or not r["path"] or type(r["allow"]) is not bool:
                raise ValueError("rule")
            by_path.setdefault(r["path"], set()).add(r["allow"])
        return {"conflicts": sorted(k for k, v in by_path.items() if len(v) > 1)}
    if case == "H36":
        p = req(payload, {"old", "new"})
        old, new = set(words(p["old"], known=KNOWN)), set(words(p["new"], known=KNOWN))
        if not set(ALWAYS) <= new or not old <= new:
            raise ValueError("weakening")
        return {"compatible": True}
    if case == "H37":
        p = req(payload, {"required", "read"})
        required, read = set(words(p["required"])), set(words(p["read"]))
        if required != read:
            raise ValueError("context")
        return {"selected": sorted(required)}
    if case == "H38":
        p = req(payload, {"paths"})
        paths = words(p["paths"])
        if any(path.startswith((".cursor/", ".github/", "harness/core/", "harness/contracts/")) or ".." in path.split("/") or path.startswith("/") for path in paths):
            raise ValueError("governance")
        return {"allowed": sorted(paths)}
    if case == "H39":
        p = req(payload, {"allowed", "refs"})
        allowed = set(words(p["allowed"]))
        refs = words(p["refs"])
        if not set(refs) <= allowed:
            raise ValueError("unauthorized")
        return {"verified": sorted(refs)}
    if case == "H40":
        p = req(payload, {"expected_sha", "observed_sha"})
        if any(not isinstance(p[k], str) or not SHA.fullmatch(p[k]) for k in p):
            raise ValueError("sha")
        if p["expected_sha"] != p["observed_sha"]:
            raise ValueError("stale")
        return {"matched": True}
    if case == "H41":
        p = req(payload, {"known", "references"})
        known = set(words(p["known"]))
        refs = set(words(p["references"]))
        return {"resolved": sorted(refs & known), "unresolved": sorted(refs - known)}
    if case == "H42":
        p = req(payload, {"project", "objects"})
        if p["project"] != "SAEBase" or not isinstance(p["objects"], list):
            raise ValueError("project")
        ids: list[str] = []
        for item in p["objects"]:
            if not isinstance(item, dict) or set(item) != {"id", "project"} or item["project"] != "SAEBase" or not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
                raise ValueError("cross_project")
            ids.append(item["id"])
        return {"ids": sorted(ids)}
    if case == "H43":
        p = req(payload, {"claims"})
        claims = p["claims"]
        if not isinstance(claims, list) or len(claims) != 2 or any(not isinstance(x, dict) or set(x) != {"source", "value"} for x in claims):
            raise ValueError("claims")
        if any(not isinstance(x["source"], str) or not x["source"] or not isinstance(x["value"], str) for x in claims) or claims[0]["source"] == claims[1]["source"]:
            raise ValueError("sources")
        return {"status": "conflict" if claims[0]["value"] != claims[1]["value"] else "consistent", "sources": sorted(x["source"] for x in claims)}
    if case == "H44":
        p = req(payload, {"comment", "actual"})
        if not all(isinstance(p[x], str) for x in p):
            raise ValueError("source")
        return {"authoritative": p["actual"], "disagreement": p["comment"] != p["actual"]}
    if case == "H45":
        p = req(payload, {"path", "exists"})
        if not isinstance(p["path"], str) or not p["path"] or type(p["exists"]) is not bool:
            raise ValueError("source")
        return {"status": "resolved" if p["exists"] else "unresolved"}
    if case == "H46":
        p = req(payload, {"claimed", "observed"})
        claimed, observed = set(words(p["claimed"])), set(words(p["observed"]))
        return {"unverified": sorted(claimed - observed), "unclaimed": sorted(observed - claimed)}
    raise ValueError("Unknown or unqualified WP19 case")
