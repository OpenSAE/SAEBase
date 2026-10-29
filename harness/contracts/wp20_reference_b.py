"""Independent reference X06-X10: actual data, SAE arithmetic, intervention and provenance."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def serial(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(value):
    return hashlib.sha256(serial(value).encode("utf-8")).hexdigest()


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def unique(xs):
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate identity")


def dot(xs, ys):
    if len(xs) != len(ys) or not all(finite(x) for x in xs + ys):
        raise ValueError("dimension or nonfinite")
    return sum(a * b for a, b in zip(xs, ys))


def x06(p, root):
    if set(p) != {"records", "selected"} or not isinstance(p["records"], list):
        raise ValueError("schema")
    records = p["records"]
    unique([r["id"] for r in records])
    if not all(r["split"] in ("train", "val", "test") and type(r["label"]) is int
               and finite(r["score"]) for r in records):
        raise ValueError("record")
    indexed = {r["id"]: r for r in records}
    unique(p["selected"])
    if not set(p["selected"]) <= set(indexed):
        raise ValueError("unknown sample")
    splits = {k: sorted(r["id"] for r in records if r["split"] == k)
              for k in ("train", "val", "test")}
    chosen = [indexed[k] for k in sorted(p["selected"])]
    predictions = [{"id": r["id"], "score": r["score"], "label": r["label"]}
                   for r in chosen]
    return {"manifest": {"sha256": sha(records), "count": len(records)},
            "splits": splits, "predictions": predictions,
            "coverage": {"evaluated": len(chosen), "total": len(records),
                         "fraction": len(chosen) / len(records) if records else 0}}


def x07(p, root):
    if set(p) != {"model", "sae", "tokens", "batch_size"}:
        raise ValueError("schema")
    m, s = p["model"], p["sae"]
    if (m["id"] != s["model_id"] or m["width"] != s["d_in"] or
            s["layer"] not in m["layers"] or p["batch_size"] < 1):
        raise ValueError("SAE identity")
    unique([r["id"] for r in p["tokens"]])
    if any(len(r["activation"]) != m["width"] or
           not all(finite(v) for v in r["activation"]) for r in p["tokens"]):
        raise ValueError("shape")
    ordered = sorted(p["tokens"], key=lambda r: r["id"])
    cache = {r["id"]: r["activation"] for r in ordered}
    identity = {"model": m["id"], "checkpoint": m["sha"], "sae": s["id"],
                "layer": s["layer"], "tokenizer": m["tokenizer"]}
    path = Path(root) / "activation-cache.json"
    path.write_text(serial({"identity": identity, "cache": cache}), encoding="utf-8")
    replay = json.loads(path.read_text(encoding="utf-8"))
    if replay["identity"] != identity or replay["cache"] != cache:
        raise ValueError("cache corrupt")
    return {"identity": identity, "samples": sorted(cache),
            "cache": cache, "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def x08(p, root):
    if set(p) != {"samples", "encoder", "decoder", "k"}:
        raise ValueError("schema")
    enc, dec = p["encoder"], p["decoder"]
    if not enc or not dec or not all(isinstance(row, list) for row in enc + dec):
        raise ValueError("weights")
    din, width = len(enc), len(enc[0])
    if not all(len(row) == width for row in enc) or len(dec) != width or not all(len(row) == din for row in dec):
        raise ValueError("matrix shape")
    k = p["k"]
    if type(k) is not int or not 0 <= k <= width:
        raise ValueError("k")
    ids = [r["id"] for r in p["samples"]]
    unique(ids)
    encoded, decoded, reconstruction = {}, {}, {}
    for row in sorted(p["samples"], key=lambda r: r["id"]):
        x = row["vector"]
        if len(x) != din:
            raise ValueError("input width")
        dense = [max(0, dot(x, [enc[i][j] for i in range(din)])) for j in range(width)]
        selected = sorted(range(width), key=lambda j: (-dense[j], j))[:k]
        sparse = [dense[j] if j in selected else 0 for j in range(width)]
        y = [dot(sparse, [dec[j][i] for j in range(width)]) for i in range(din)]
        encoded[row["id"]] = {"active_ids": selected, "values": sparse}
        decoded[row["id"]] = y
        reconstruction[row["id"]] = sum((a - b) ** 2 for a, b in zip(x, y)) / din
    return {"shape": {"d_in": din, "features": width}, "encoded": encoded,
            "decoded": decoded, "mse": reconstruction}


def x09(p, root):
    if set(p) != {"activations", "steps", "mode"} or p["mode"] not in ("teacher_forced", "generation"):
        raise ValueError("schema")
    values = {r["id"]: list(r["features"]) for r in p["activations"]}
    unique(list(values))
    baseline = json.loads(serial(values))
    trace = []
    for s in p["steps"]:
        sid, index, kind = s["sample"], s["feature"], s["op"]
        if sid not in values or type(index) is not int or not 0 <= index < len(values[sid]):
            raise ValueError("invalid target")
        arr = values[sid]
        before = arr[index]
        if kind == "ablate":
            arr[index] = 0
        elif kind == "add":
            arr[index] += s["value"]
        elif kind == "scale":
            arr[index] *= s["value"]
        elif kind == "clamp":
            arr[index] = min(max(arr[index], s["low"]), s["high"])
        else:
            raise ValueError("operator")
        if not finite(arr[index]):
            raise ValueError("nonfinite")
        trace.append({"sample": sid, "feature": index, "op": kind,
                      "before": before, "after": arr[index]})
    result = json.loads(serial(values))
    return {"mode": p["mode"], "baseline": baseline, "trace": trace,
            "after": result, "rollback": baseline}


def x10(p, root):
    if set(p) != {"tokens", "features", "topk"}:
        raise ValueError("schema")
    ids = [r["id"] for r in p["tokens"]]
    unique(ids)
    width = p["features"]
    if type(width) is not int or width < 1 or not 1 <= p["topk"] <= width:
        raise ValueError("dimensions")
    scored = {}
    provenance = []
    for token in p["tokens"]:
        activations = token["activations"]
        if len(activations) != width or not all(finite(v) for v in activations):
            raise ValueError("activation")
        selected = sorted(range(width), key=lambda i: (-activations[i], i))[:p["topk"]]
        scored[token["id"]] = selected
        for feature in selected:
            provenance.append({"token": token["id"], "feature": feature,
                               "value": activations[feature]})
    return {"token_ids": sorted(ids), "ranked_ids": scored,
            "provenance": sorted(provenance, key=lambda r: (r["token"], r["feature"])),
            "unresolved": []}


CASES = {"X06": x06, "X07": x07, "X08": x08, "X09": x09, "X10": x10}
