from __future__ import annotations

import hashlib
import json
import math


def require_keys(payload, keys):
    if set(payload) != set(keys):
        raise ValueError("schema")


def unique(values):
    if len(set(values)) != len(values):
        raise ValueError("duplicate")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serial(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(value):
    return hashlib.sha256(serial(value).encode("utf-8")).hexdigest()


def dot(left, right):
    if len(left) != len(right) or not all(finite(x) for x in left + right):
        raise ValueError("dimension or nonfinite")
    return sum(a * b for a, b in zip(left, right))
