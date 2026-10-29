"""Schedule synthetic CI jobs in a stable topological order."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a job id is duplicated, a dependency is missing, or the graph cycles."""


def solve(payload: dict) -> dict:
    """Return a deterministic job order. Ties break by job id."""
    if not isinstance(payload, dict) or set(payload) != {"jobs"}:
        raise InputError("schema")
    return {"order": _schedule(payload["jobs"])}


def _schedule(values: object) -> list[str]:
    """Kahn sort. A job is ready only after every named dependency has run."""
    if not isinstance(values, list):
        raise InputError("jobs")
    needs: dict[str, list[str]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"id", "needs"}:
            raise InputError("job")
        job_id = value["id"]
        deps = value["needs"]
        if not isinstance(job_id, str) or not job_id or job_id in needs:
            raise InputError("id")
        if not isinstance(deps, list):
            raise InputError("needs")
        clean: list[str] = []
        seen: set[str] = set()
        for dep in deps:
            if not isinstance(dep, str) or not dep or dep in seen:
                raise InputError("needs")
            seen.add(dep)
            clean.append(dep)
        needs[job_id] = clean
    for deps in needs.values():
        if any(dep not in needs for dep in deps):
            raise InputError("missing")
    pending = {job_id: set(deps) for job_id, deps in needs.items()}
    ready = sorted(job_id for job_id, deps in pending.items() if not deps)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for job_id, deps in pending.items():
            if current in deps:
                deps.remove(current)
                if not deps and job_id not in order and job_id not in ready:
                    ready.append(job_id)
        ready.sort()
    if len(order) != len(needs):
        raise InputError("cycle")
    return order
