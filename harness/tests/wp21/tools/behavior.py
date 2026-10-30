"""Reusable, production-output assertions for independent WP21 reviewers.

These functions operate on observed numerical values, subprocess results, and
real output files. They do not replace running the production SAEBase model,
registry, trainer, search or experiment code in a fresh reviewer process.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

from common import canonical_hash, contained, sha256


def finite_vector(values: list[float], *, name: str = "vector") -> list[float]:
    if not isinstance(values, list) or not values:
        raise AssertionError(f"{name}: nonempty numeric vector required")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise AssertionError(f"{name}: non-finite or non-numeric entry")
    return [float(v) for v in values]


def close_vectors(actual: list[float], expected: list[float], *, rtol: float = 1e-4, atol: float = 1e-4) -> float:
    a, b = finite_vector(actual, name="actual"), finite_vector(expected, name="expected")
    if len(a) != len(b):
        raise AssertionError("vector shape mismatch; broadcasting forbidden")
    if rtol < 0 or atol < 0:
        raise ValueError("nonnegative numerical tolerances required")
    maximum = max(abs(x - y) for x, y in zip(a, b))
    if any(abs(x - y) > atol + rtol * abs(y) for x, y in zip(a, b)):
        raise AssertionError(f"numerical tolerance exceeded: max_abs={maximum:g}")
    return maximum


def reconstruction_metrics(activation: list[float], reconstruction: list[float], *, measured_error: float, measured_norm: float) -> dict:
    a, r = finite_vector(activation, name="activation"), finite_vector(reconstruction, name="reconstruction")
    if len(a) != len(r):
        raise AssertionError("reconstruction shape mismatch")
    error = math.sqrt(sum((x-y)**2 for x, y in zip(a, r)))
    norm = math.sqrt(sum(y*y for y in r))
    close_vectors([measured_error,measured_norm],[error,norm], rtol=1e-5, atol=1e-5)
    return {"error_l2":error, "reconstruction_l2":norm}


def mse_metrics(activation: list[float], reconstruction: list[float], *, measured_mse: float) -> dict:
    a, r = finite_vector(activation), finite_vector(reconstruction)
    if len(a) != len(r):
        raise AssertionError("MSE vectors require matching shapes")
    mse = sum((x-y)**2 for x,y in zip(a,r))/len(a)
    close_vectors([measured_mse],[mse],rtol=1e-5,atol=1e-5)
    return {"mse":mse}


def intervention_invariants(*, old_codes: list[float], new_codes: list[float], replacements: dict[int, float],
                            original_activation: list[float], edited_activation: list[float],
                            old_reconstruction: list[float], new_reconstruction: list[float], rtol: float=1e-4, atol: float=1e-4) -> dict:
    old, new = finite_vector(old_codes), finite_vector(new_codes)
    if len(old) != len(new):
        raise AssertionError("codes width changed")
    expected = old[:]
    for index, value in replacements.items():
        if isinstance(index,bool) or not isinstance(index,int) or index<0 or index>=len(old):
            raise AssertionError("out of bounds replacement index")
        if isinstance(value,bool) or not math.isfinite(value):
            raise AssertionError("invalid replacement value")
        expected[index] = float(value)
    close_vectors(new,expected,rtol=rtol,atol=atol)
    act, edit = finite_vector(original_activation), finite_vector(edited_activation)
    old_r, new_r = finite_vector(old_reconstruction), finite_vector(new_reconstruction)
    if not (len(act)==len(edit)==len(old_r)==len(new_r)):
        raise AssertionError("activation/reconstruction shape mismatch")
    close_vectors(edit,[a + (r1-r0) for a,r0,r1 in zip(act,old_r,new_r)],rtol=rtol,atol=atol)
    return {"width":len(new),"changed_features":len(replacements),"activation_width":len(act)}


def component_lifecycle(before: list[str], after: list[str], *, retired: list[str], retained: list[str]) -> dict:
    for label, values in [("before",before),("after",after)]:
        if not isinstance(values,list) or len(values)!=len(set(values)):
            raise AssertionError(label+" must be unique observed registry IDs")
    if not set(retired)<=set(before):
        raise AssertionError("removed ID was never installed in before process")
    if set(retired)&set(after):
        raise AssertionError("removed ID remains discoverable in fresh process")
    if not set(retained)<=set(before)&set(after):
        raise AssertionError("required unaffected component absent")
    return {"retired":len(retired),"retained":len(retained)}


def grid_complete(*, grid: dict[str, list], trials: list[dict], require_success: bool=True) -> dict:
    if not grid or not isinstance(grid,dict) or any(not isinstance(x,list) or not x for x in grid.values()):
        raise AssertionError("nonempty grid values required")
    keys=list(grid)
    combos=[dict(zip(keys,values)) for values in itertools.product(*(grid[k] for k in keys))]
    expected={canonical_hash(c) for c in combos}
    if len(expected)!=len(combos):
        raise AssertionError("duplicate grid combinations")
    observed=[]
    ids=[]
    for t in trials:
        if not isinstance(t,dict) or not all(k in t for k in ("trial_id","parameters","status")):
            raise AssertionError("trial identity/status missing")
        ids.append(t["trial_id"])
        observed.append(canonical_hash(t["parameters"]))
        if require_success and (t["status"]!="completed" or t.get("exit_code")!=0):
            raise AssertionError("trial is missing true successful execution")
    if len(set(ids))!=len(ids) or len(set(observed))!=len(observed) or set(observed)!=expected:
        raise AssertionError("missing/duplicate/unplanned trial combination or ID")
    return {"grid_size":len(combos),"trials":len(trials)}


def argmin_mean(*, trials: list[dict], group_by: str, value_key: str, tie_break: list[str]) -> dict:
    grouped={}
    for t in trials:
        label=t[group_by]; value=t[value_key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise AssertionError("missing/nonfinite optimization metric")
        grouped.setdefault(label,[]).append(value)
    if not grouped:
        raise AssertionError("no successful trials")
    if set(grouped) != set(tie_break):
        raise AssertionError("selection candidates disagree with frozen order")
    scores={key:sum(v)/len(v) for key,v in grouped.items()}
    best=min(tie_break,key=lambda key:(scores[key],tie_break.index(key)))
    return {"best":best, "means":scores}


def file_manifest(root: Path, entries: list[dict], *, allow_unlisted: bool=False) -> dict:
    if not isinstance(entries,list) or not entries:
        raise AssertionError("nonempty manifest required")
    listed=set()
    total=0
    root=root.resolve(strict=True)
    for entry in entries:
        rel=entry["path"]
        if rel in listed:
            raise AssertionError("duplicate manifest path")
        listed.add(rel)
        p=contained(root,rel,exists=True)
        if not p.is_file() or p.stat().st_size != entry["size_bytes"] or sha256(p) != entry["sha256"]:
            raise AssertionError("manifest byte/hash discrepancy: "+rel)
        total += p.stat().st_size
    if not allow_unlisted:
        actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
        if actual!=listed:
            raise AssertionError("manifest misses or adds entries")
    return {"files":len(listed),"size_bytes":total}


def resume_integrity(before: list[dict], after: list[dict], *, reused: list[str], identity_before: dict, identity_after: dict) -> dict:
    if not isinstance(identity_before,dict) or not identity_before or identity_before!=identity_after:
        raise AssertionError("resume identities disagree")
    prior={x["path"]:x["sha256"] for x in before}
    current={x["path"]:x["sha256"] for x in after}
    if len(prior)!=len(before) or len(current)!=len(after):
        raise AssertionError("duplicate artifact paths")
    for path in reused:
        if path not in prior or current.get(path)!=prior[path]:
            raise AssertionError("reused successful artifact changed")
    if not set(prior)<=set(current):
        raise AssertionError("older evidence was erased")
    return {"reused":len(reused),"new":len(set(current)-set(prior))}


def pipeline_integrity(root: Path, stages: list[dict]) -> dict:
    if not isinstance(stages,list) or not stages:
        raise AssertionError("pipeline stages required")
    seen={}
    for s in stages:
        sid=s["id"]
        if sid in seen:
            raise AssertionError("duplicate stage ID")
        status=s["status"]
        for parent in s.get("parents",[]):
            if parent not in seen or seen[parent]["status"]!="completed":
                if status=="completed":
                    raise AssertionError("completed stage has absent/failed parent")
        if status=="completed":
            path=contained(root,s["path"],exists=True)
            if not path.is_file() or sha256(path)!=s["sha256"]:
                raise AssertionError("completed stage bytes changed or missing")
        seen[sid]=s
    return {"stages":len(stages),"completed":sum(s["status"]=="completed" for s in stages)}


def observed_device(*, configured: str | None, actual: str,
                    requires_cuda: bool = False) -> dict:
    """Report a genuine runtime device instead of trusting a YAML preference.

    Optional for inference tests: the frozen R12/R13 tasks allow documenting
    an existing CPU fallback. Training R17 requires an actual CUDA device.
    """
    if not isinstance(actual, str) or not actual.strip():
        raise AssertionError("model or activation actual device was not observed")
    prefix = actual.split(":", 1)[0]
    if prefix not in {"cpu", "cuda", "mps"}:
        raise AssertionError("invalid observed device")
    if configured is not None and configured.split(":", 1)[0] not in {"cpu", "cuda", "mps"}:
        raise AssertionError("unsupported configured device")
    if requires_cuda and prefix != "cuda":
        raise AssertionError("CUDA-required task actually executed elsewhere")
    return {"actual_device":actual, "configured_device":configured,
            "configured_matches_actual":configured is None or configured.split(":",1)[0] == prefix,
            "cuda_required":requires_cuda}
