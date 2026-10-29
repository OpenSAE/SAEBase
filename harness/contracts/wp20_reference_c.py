"""Independent X11-X15: metrics, seeded resampling, adapters, config and recovery."""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def serial(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj):
    return hashlib.sha256(serial(obj).encode("utf-8")).hexdigest()


def mean(values):
    if not values or not all(finite(x) for x in values):
        raise ValueError("invalid mean")
    return sum(values) / len(values)


def x11(p, root):
    if set(p) != {"labels", "scores", "threshold"}:
        raise ValueError("schema")
    labels, scores, thr = p["labels"], p["scores"], p["threshold"]
    if (not labels or len(labels) != len(scores) or any(type(y) is not int or y not in (0, 1) for y in labels)
            or not all(finite(x) for x in scores) or not finite(thr)):
        raise ValueError("labels or scores")
    positives = [scores[i] for i, y in enumerate(labels) if y == 1]
    negatives = [scores[i] for i, y in enumerate(labels) if y == 0]
    if not positives or not negatives:
        raise ValueError("AUROC undefined for a single class")
    auc = sum(1 if p > n else 0.5 if p == n else 0 for p in positives for n in negatives) / (len(positives) * len(negatives))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    groups = []
    tp = fp = ap = prev_recall = 0
    for score in sorted(set(scores), reverse=True):
        group = [i for i in order if scores[i] == score]
        tp += sum(labels[i] for i in group)
        fp += sum(1-labels[i] for i in group)
        recall = tp/len(positives)
        precision = tp/(tp+fp)
        ap += (recall-prev_recall)*precision
        prev_recall = recall
        groups.append({"score": score, "tp": tp, "fp": fp,
                       "precision": precision, "recall": recall})
    total_tp = sum(y == 1 and score >= thr for y,score in zip(labels,scores))
    return {"n": len(labels), "auc": auc, "average_precision": ap,
            "recall": total_tp/len(positives), "threshold": thr, "curve": groups}


def x12(p, root):
    if set(p) != {"groups", "seed", "draws"} or type(p["seed"]) is not int or type(p["draws"]) is not int or not 1 <= p["draws"] <= 60:
        raise ValueError("schema")
    groups = p["groups"]
    ids = [x["id"] for x in groups]
    if len(set(ids)) != len(ids) or not groups or any(not x["values"] or not all(finite(v) for v in x["values"]) for x in groups):
        raise ValueError("groups")
    ordered = sorted(groups, key=lambda r:r["id"])
    rng = random.Random(p["seed"])
    samples = []
    stats = []
    for _ in range(p["draws"]):
        selected = [rng.randrange(len(ordered)) for __ in ordered]
        samples.append([ordered[i]["id"] for i in selected])
        stats.append(mean([v for i in selected for v in ordered[i]["values"]]))
    sorted_stats = sorted(stats)
    return {"baseline": mean([v for r in ordered for v in r["values"]]),
            "replicate_indices": samples, "replicate_means": stats,
            "interval": [sorted_stats[int(.025*(len(stats)-1))],
                         sorted_stats[int(.975*(len(stats)-1))]],
            "seed": p["seed"]}


def x13(p, root):
    if set(p) != {"adapters"}:
        raise ValueError("schema")
    ids=[a["id"] for a in p["adapters"]]
    if len(set(ids))!=len(ids) or not ids:
        raise ValueError("duplicate or empty adapters")
    details=[];valid=[]
    for adapter in p["adapters"]:
        rows=adapter["rows"]
        if not isinstance(rows,list) or len({r["id"] for r in rows})!=len(rows):
            raise ValueError("sample identity")
        successful=[r for r in rows if r["status"]=="success"]
        if any(not finite(r["value"]) for r in successful):
            raise ValueError("numeric metric")
        if any(r["status"] not in ("success","failed") for r in rows):
            raise ValueError("status")
        details.append({"id":adapter["id"],"n":len(rows),"valid":len(successful),
                        "mean":mean([r["value"] for r in successful]) if successful else None})
        valid.extend(r["value"] for r in successful)
    return {"adapters":sorted(details,key=lambda r:r["id"]),
            "overall_mean":mean(valid) if valid else None,
            "evaluated":len(valid),"total":sum(r["n"] for r in details)}


def x14(p, root):
    if set(p)!={"base","overrides","models","datasets","budget"}:
        raise ValueError("schema")
    cfg=dict(p["base"])
    if set(cfg)!= {"model","dataset","seed"} or any(k not in cfg for k in p["overrides"]):
        raise ValueError("config")
    cfg.update(p["overrides"])
    if (cfg["model"] not in p["models"] or cfg["dataset"] not in p["datasets"]
            or type(cfg["seed"]) is not int or type(p["budget"]) is not int or not 0<=p["budget"]<=4):
        raise ValueError("selection")
    jobs=[{"id":f"job-{i}", "model":cfg["model"],"dataset":cfg["dataset"],
           "seed":cfg["seed"]+i} for i in range(p["budget"])]
    return {"resolved":cfg, "config_sha256":sha(cfg),
            "jobs":jobs,"dry_run":True,"receipt":{"planned":len(jobs),"executed":0}}


def x15(p, root):
    if set(p)!={"identity","stages","interrupt_after","resume_identity"}:
        raise ValueError("schema")
    identity=p["identity"]
    if (not isinstance(identity,dict) or set(identity)!= {"run","seed","config_sha","data_sha"}
            or identity!=p["resume_identity"] or type(identity["seed"]) is not int):
        raise ValueError("checkpoint mismatch")
    stage_ids=[s["id"] for s in p["stages"]]
    if len(set(stage_ids))!=len(stage_ids) or type(p["interrupt_after"]) is not int or not 0<=p["interrupt_after"]<=len(stage_ids):
        raise ValueError("stages")
    path=Path(root)/"checkpoint.json"
    stages=[{"id":s["id"],"value":s["value"]} for s in p["stages"]]
    partial=stages[:p["interrupt_after"]]
    path.write_text(serial({"identity":identity,"completed":partial}),encoding="utf-8")
    checkpoint=json.loads(path.read_text(encoding="utf-8"))
    remaining=stages[len(checkpoint["completed"]):]
    completed=checkpoint["completed"]+remaining
    path.write_text(serial({"identity":identity,"completed":completed}),encoding="utf-8")
    repeat=json.loads(path.read_text(encoding="utf-8"))
    if repeat["completed"]!=stages:
        raise ValueError("non-idempotent")
    return {"checkpoint_sha256":sha({"identity":identity,"completed":partial}),
            "interrupted_at":p["interrupt_after"],"completed":completed,
            "resumed":len(remaining),"idempotent":repeat["completed"]==stages}


CASES={"X11":x11,"X12":x12,"X13":x13,"X14":x14,"X15":x15}
