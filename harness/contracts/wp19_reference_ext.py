"""Independent reference semantics for WP19 synthetic invariant exercises H47-H226.

These are *bounded behavior contracts*, not assertions that real product code
has been integrated, executed on GPU, or validated on research data.
"""

from __future__ import annotations

import math
import re


def _path(p):
    return (
        isinstance(p, str)
        and bool(p)
        and not p.startswith(("/", "~"))
        and "\\" not in p
        and ":" not in p
        and all(x not in ("", ".", "..") for x in p.split("/"))
    )


def _unique(xs):
    return (
        isinstance(xs, list)
        and all(isinstance(x, str) and x for x in xs)
        and len(set(xs)) == len(xs)
    )


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _valid(op, v):
    if not isinstance(v, dict):
        return False
    if op == "relative":
        return _path(v.get("path"))
    if op == "protected":
        return _path(v.get("path")) and not v["path"].startswith(
            ("harness/", "../", ".cursor/", ".github/")
        )
    if op == "unique":
        return _unique(v.get("ids"))
    if op == "subset":
        return (
            _unique(v.get("allowed"))
            and _unique(v.get("selected"))
            and set(v["selected"]) <= set(v["allowed"])
        )
    if op == "equal":
        return type(v.get("expected")) is str and v["expected"] == v.get("observed")
    if op == "different":
        return (
            type(v.get("before")) is str
            and type(v.get("after")) is str
            and v["before"] != v["after"]
        )
    if op == "digest":
        return (
            isinstance(v.get("expected"), str)
            and re.fullmatch(r"[0-9a-f]{64}", v["expected"]) is not None
            and v["expected"] == v.get("observed")
        )
    if op == "approved":
        return v.get("approved") is True and v.get("actor") in ("reviewer", "human")
    if op == "unknown":
        return type(v.get("synthetic")) is bool and v.get("observed") is False
    if op in ("budget", "timeout"):
        a, b = (v.get("limit"), v.get("used" if op == "budget" else "elapsed"))
        return _finite(a) and _finite(b) and 0 <= b <= a
    if op == "finite":
        return _finite(v.get("value"))
    if op == "ordered":
        return _unique(v.get("order")) and v["order"] == sorted(v["order"])
    if op == "scope":
        return (
            _unique(v.get("nodes"))
            and isinstance(v.get("target"), str)
            and v["target"] in v["nodes"]
        )
    if op == "reciprocal":
        a, b = v.get("forward"), v.get("backward")
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        return all(
            isinstance(xs, list) and all(k in b and u in b[k] for k in xs) for u, xs in a.items()
        ) and all(
            isinstance(xs, list) and all(u in a and k in a[u] for u in xs) for k, xs in b.items()
        )
    if op == "locale":
        a, b = v.get("zh"), v.get("en")
        return (
            isinstance(a, str)
            and isinstance(b, str)
            and bool(a.strip())
            and bool(b.strip())
            and "翻译不可用" not in a
            and "translation unavailable" not in b.lower()
        )
    if op == "redacted":

        def walk(x, key="", depth=0):
            if depth > 24:
                return False
            if isinstance(x, dict):
                return all(isinstance(k, str) and walk(val, k, depth + 1) for k, val in x.items())
            if isinstance(x, list):
                return all(walk(z, key, depth + 1) for z in x)
            if isinstance(x, str):
                if re.search(r"(?i)token|secret|cookie|password|private.?key|credential", key):
                    return x == "[REDACTED]"
                return re.search(r"(?i)(token|secret|password)\s*[:=]\s*\S+", x) is None
            return x is None or type(x) in (bool, int, float)

        return walk(v.get("record"))
    if op == "revision":
        return (
            type(v.get("expected")) is int
            and v["expected"] >= 0
            and type(v.get("actual")) is int
            and v["expected"] == v["actual"]
        )
    if op == "immutable":
        return (
            "before" in v
            and "after" in v
            and type(v["before"]) is type(v["after"])
            and v["before"] == v["after"]
        )
    if op == "transition":
        return v.get("authorized") is True and (v.get("previous"), v.get("next")) in (
            ("unchecked", "approve"),
            ("deny", "unchecked"),
            ("unchecked", "deny"),
            ("approve", "unchecked"),
        )
    if op in ("edges", "topology"):
        nodes = v.get("nodes")
        edges = v.get("edges")
        if (
            not _unique(nodes)
            or not isinstance(edges, list)
            or any(
                not isinstance(e, list) or len(e) != 2 or any(x not in nodes for x in e)
                for e in edges
            )
        ):
            return False
        if op == "edges":
            return True
        order = v.get("order")
        return (
            _unique(order)
            and set(order) == set(nodes)
            and all(order.index(a) < order.index(b) for a, b in edges)
        )
    if op == "tree":
        rows = v.get("nodes")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows
        ):
            return False
        ids = [r["id"] for r in rows]
        if not _unique(ids):
            return False
        parents = {r["id"]: r.get("parent") for r in rows}
        for id_ in ids:
            seen = set()
            cur = id_
            while cur is not None:
                if cur in seen or cur not in parents:
                    return False
                seen.add(cur)
                cur = parents[cur]
        return True
    if op in ("mae", "recall", "ratio"):
        if op == "mae":
            a, b, m = v.get("pred"), v.get("target"), v.get("mae")
            return (
                isinstance(a, list)
                and isinstance(b, list)
                and bool(a)
                and len(a) == len(b)
                and all(_finite(x) for x in a + b)
                and _finite(m)
                and math.isclose(sum(abs(x - y) for x, y in zip(a, b)) / len(a), m, abs_tol=1e-9)
            )
        if op == "recall":
            tp, fn, r = v.get("tp"), v.get("fn"), v.get("recall")
            return (
                type(tp) is int
                and type(fn) is int
                and tp >= 0
                and fn >= 0
                and tp + fn > 0
                and _finite(r)
                and math.isclose(tp / (tp + fn), r, abs_tol=1e-9)
            )
        a, b, c = v.get("done"), v.get("total"), v.get("coverage")
        return (
            type(a) is int
            and type(b) is int
            and b > 0
            and 0 <= a <= b
            and _finite(c)
            and math.isclose(a / b, c, abs_tol=1e-9)
        )
    if op == "vector":
        before, after, idx, mode, delta = (
            v.get("before"),
            v.get("after"),
            v.get("index"),
            v.get("mode"),
            v.get("delta"),
        )
        if (
            not isinstance(before, list)
            or not isinstance(after, list)
            or len(before) != len(after)
            or type(idx) is not int
            or idx < 0
            or idx >= len(before)
            or not all(_finite(x) for x in before + after)
            or not _finite(delta)
        ):
            return False
        ref = list(before)
        if mode == "ablate":
            ref[idx] = 0
        elif mode == "scale":
            ref[idx] *= delta
        elif mode == "add":
            ref[idx] += delta
        elif mode == "clamp":
            ref[idx] = min(max(ref[idx], 0), delta)
        else:
            return False
        return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(ref, after))
    if op == "topk":
        s, k, chosen = v.get("scores"), v.get("k"), v.get("chosen")
        return (
            isinstance(s, list)
            and all(_finite(x) for x in s)
            and type(k) is int
            and 0 <= k <= len(s)
            and isinstance(chosen, list)
            and chosen == sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
        )
    if op == "matrix":
        a, b = v.get("a"), v.get("b")
        return (
            isinstance(a, list)
            and bool(a)
            and isinstance(b, list)
            and bool(b)
            and all(isinstance(row, list) and row and all(_finite(x) for x in row) for row in a + b)
            and len(set(map(len, a))) == 1
            and len(set(map(len, b))) == 1
            and len(a[0]) == len(b)
        )
    if op == "splits":
        return (
            _unique(v.get("train"))
            and _unique(v.get("test"))
            and not set(v["train"]) & set(v["test"])
        )
    if op == "resource":
        return (
            type(v.get("gpu_minutes")) is int
            and v["gpu_minutes"] >= 0
            and type(v.get("authorized")) is bool
            and (v["gpu_minutes"] == 0 or v["authorized"])
        )
    if op == "release":
        return (
            v.get("approved") is True
            and isinstance(v.get("checks"), list)
            and bool(v["checks"])
            and all(x == "success" for x in v["checks"])
        )
    if op == "idempotent":
        return (
            "first" in v
            and "retry" in v
            and v["first"] == v["retry"]
            and type(v["first"]) is type(v["retry"])
        )
    if op == "version":
        return (
            isinstance(v.get("source"), str)
            and bool(v["source"])
            and v["source"] == v.get("cache_source")
        )
    if op == "archive":
        return _unique(v.get("members")) and all(_path(x) for x in v["members"])
    if op == "manifest":
        files = v.get("files")
        if not isinstance(files, list):
            return False
        ids = []
        for row in files:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not row["id"]
                or not isinstance(row.get("sha"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha"]) is None
            ):
                return False
            ids.append(row["id"])
        return len(ids) == len(set(ids))
    if op == "namespace":
        value = v.get("module")
        return (
            isinstance(value, str)
            and re.fullmatch(r"(tasks|models|datasets)\.[a-z_][a-z_0-9]*", value) is not None
        )
    if op == "numbers":
        return isinstance(v.get("values"), list) and all(_finite(x) for x in v["values"])
    if op == "provenance":
        return (
            type(v.get("resolved")) is bool
            and type(v.get("exists")) is bool
            and v["resolved"] == v["exists"]
        )
    if op == "schema":
        a, b = v.get("required"), v.get("record")
        return _unique(a) and isinstance(b, dict) and set(a) <= set(b)
    if op == "allowed_paths":
        scope, p = v.get("scope"), v.get("path")
        return (
            _unique(scope)
            and all(isinstance(x, str) for x in scope)
            and _path(p)
            and any(p.startswith(s) for s in scope)
        )
    if op == "command":
        return (
            _unique(v.get("allowlist"))
            and isinstance(v.get("command"), str)
            and v["command"] in v["allowlist"]
        )
    if op == "pairing":
        return (
            _unique(v.get("left")) and _unique(v.get("right")) and set(v["left"]) == set(v["right"])
        )
    raise ValueError("Unqualified operator")


def solve(case, payload, spec):
    if not isinstance(payload, dict) or set(payload) != {"subject"}:
        raise ValueError("Input must contain one subject")
    if (
        not isinstance(spec, dict)
        or spec.get("id") != case
        or not isinstance(spec.get("operator"), str)
    ):
        raise ValueError("Unqualified case")
    if not _valid(spec["operator"], payload["subject"]):
        raise ValueError("Contract violated")
    return {"accepted": True}
