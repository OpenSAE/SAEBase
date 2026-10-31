"""Read frozen experiment assets without silently substituting another source.

The researcher-published project lock is distinct from a first-run resolved
local identity. A first-run identity supports same-machine resume, not
cross-machine scientific equivalence.
"""

from __future__ import annotations


from experiments.qa500.config import get_assets_lock
from saebase.integrations.hf_assets import resolve_revision


def declared_revision(kind: str, name: str) -> str | None:
    key = {
        "dataset": "dataset_revisions",
        "model": "model_revisions",
        "sae": "sae_revisions",
    }.get(kind)
    if key is None:
        raise ValueError("Unknown asset type")
    value = get_assets_lock().get(key, {}).get(name)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError("Malformed experiment asset lock")
    return value


def source_revision(repo_id: str, key: str) -> tuple[str, str]:
    expected = declared_revision("dataset", key)
    if expected is not None:
        return expected, "project_locked"
    return resolve_revision(repo_id, repo_type="dataset"), "first_run_frozen"


def required_publication_lock() -> bool:
    lock = get_assets_lock()
    return lock.get("status") == "pinned" and all(
        lock.get(key) for key in ("dataset_revisions", "model_revisions", "sae_revisions")
    )
