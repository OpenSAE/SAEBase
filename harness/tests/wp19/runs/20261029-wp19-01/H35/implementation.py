"""Report paths that are both allowed and denied in the same rule list."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when a rule is missing a nonempty path or a real boolean allow flag."""


def solve(payload: dict) -> dict:
    """Return conflicting paths in dictionary order. Same-path same-flag is not a conflict."""
    if not isinstance(payload, dict) or set(payload) != {"rules"}:
        raise InputError("schema")
    rules = payload["rules"]
    if not isinstance(rules, list):
        raise InputError("rules")
    by_path: dict[str, set[bool]] = {}
    for rule in rules:
        path, allow = _rule(rule)
        by_path.setdefault(path, set()).add(allow)
    return {"conflicts": sorted(path for path, flags in by_path.items() if len(flags) > 1)}


def _rule(rule: object) -> tuple[str, bool]:
    """Read one path and one boolean allow flag."""
    if not isinstance(rule, dict) or set(rule) != {"path", "allow"}:
        raise InputError("rule")
    path = rule["path"]
    allow = rule["allow"]
    if not isinstance(path, str) or not path:
        raise InputError("path")
    if type(allow) is not bool:
        raise InputError("allow")
    return path, allow
