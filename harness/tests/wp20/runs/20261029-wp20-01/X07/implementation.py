from __future__ import annotations

from pathlib import Path

from src.pipeline import execute


class InputError(ValueError):
    """Illegal WP20 payload for this case."""


def run(payload: dict, workdir: Path) -> dict:
    if not isinstance(payload, dict):
        raise InputError("payload")
    try:
        return execute(payload, Path(workdir))
    except ValueError as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(str(exc)) from exc
