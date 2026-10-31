"""Read frozen experiment assets without silently substituting another source.

The researcher-published project lock is distinct from a first-run resolved
local identity. A first-run identity supports same-machine resume, not
cross-machine scientific equivalence.
"""

from __future__ import annotations


import re

from experiments.qa500.config import get_assets_lock, get_dataset_ids, get_sites
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
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Malformed experiment asset lock")
    return value


def source_revision(repo_id: str, key: str) -> tuple[str, str]:
    expected = declared_revision("dataset", key)
    if expected is not None:
        return expected, "project_locked"
    return resolve_revision(repo_id, repo_type="dataset"), "first_run_frozen"


def required_publication_lock() -> bool:
    """Check full study asset coverage, never merely nonempty lock dictionaries.

    Keys are protocol dataset IDs, model registry keys, and the exact
    `model_key:sae_key:L{layer}` SAE sites. Values are full upstream commits.
    A future release must ALSO compare these revisions with downloaded assets
    and independently verified real run manifests.
    """
    lock = get_assets_lock()
    if lock.get("status") != "pinned":
        return False
    sites = get_sites()
    expected = {
        "dataset_revisions": set(get_dataset_ids()),
        "model_revisions": {entry["model_key"] for entry in sites},
        "sae_revisions": {
            f"{entry['model_key']}:{entry['sae_key']}:L{layer}"
            for entry in sites
            for layer in entry["layers"]
        },
    }
    for field, required_keys in expected.items():
        values = lock.get(field)
        if not isinstance(values, dict) or set(values) != required_keys:
            return False
        if any(
            not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None
            for sha in values.values()
        ):
            return False
    return True
