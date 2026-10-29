"""Redact nested synthetic credentials and private paths."""

from __future__ import annotations

_SECRET = {
    "access_token",
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "cookie",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "session",
    "token",
}
_MAX_DEPTH = 8


class InputError(ValueError):
    """Raised when a payload cannot be fully inspected. The message holds no caller value."""


def solve(payload: dict) -> dict:
    """Return a redacted copy of the object stored under payload."""
    if not isinstance(payload, dict) or set(payload) != {"payload"}:
        raise InputError("schema")
    body = payload["payload"]
    if not isinstance(body, dict):
        raise InputError("payload")
    return {"payload": _redact(body, 0)}


def _redact(value: object, depth: int) -> object:
    """Walk dicts and lists. Unsupported shapes fail instead of passing through."""
    if depth > _MAX_DEPTH:
        raise InputError("depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise InputError("number")
        return value
    if isinstance(value, str):
        if value.startswith("/") or value.startswith("~/") or "\\" in value:
            return "[REDACTED]"
        return value
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputError("key")
            if key.casefold() in _SECRET:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item, depth + 1)
        return redacted
    raise InputError("type")
