"""Keep SAEBase objects inside SAEBase. A foreign project is not renamed."""

from __future__ import annotations


class InputError(ValueError):
    """Raised for a non-SAEBase project, a duplicate id, or a malformed object."""


def solve(payload: dict) -> dict:
    """Return object ids in dictionary order when every object belongs to SAEBase."""
    if not isinstance(payload, dict) or set(payload) != {"project", "objects"}:
        raise InputError("schema")
    if payload["project"] != "SAEBase":
        raise InputError("project")
    objects = payload["objects"]
    if not isinstance(objects, list):
        raise InputError("objects")
    ids: list[str] = []
    for item in objects:
        object_id = _object(item)
        if object_id in ids:
            raise InputError("duplicate")
        ids.append(object_id)
    return {"ids": sorted(ids)}


def _object(item: object) -> str:
    """Read one SAEBase object with a nonempty id."""
    if not isinstance(item, dict) or set(item) != {"id", "project"}:
        raise InputError("object")
    if item["project"] != "SAEBase":
        raise InputError("cross_project")
    object_id = item["id"]
    if not isinstance(object_id, str) or not object_id:
        raise InputError("id")
    return object_id
