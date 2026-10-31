"""Versioned Hugging Face asset discovery for any SAEBase experiment.

Network requests are explicit. No substitute checkpoint or unpinned success
is invented on a failed download. An approved lock can be used offline.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Literal


_REV = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HFAsset:
    repo_id: str
    repo_type: Literal["model", "dataset"]
    revision: str

    def validate(self) -> None:
        if not self.repo_id or not _REV.fullmatch(self.revision):
            raise ValueError("Asset must have an exact 40-hex Hugging Face revision")
        if self.repo_type not in {"model", "dataset"}:
            raise ValueError("Unknown asset family")


def resolve_revision(repo_id: str, *, repo_type: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "dataset":
        revision = api.dataset_info(repo_id).sha
    elif repo_type == "model":
        revision = api.model_info(repo_id).sha
    else:
        raise ValueError("Unsupported HF repository kind")
    if not isinstance(revision, str) or not _REV.fullmatch(revision):
        raise RuntimeError("HF source identity is not a full revision hash")
    return revision


def download(asset: HFAsset, *, cache_dir: Path | None = None) -> Path:
    """Download a *specified* source revision using official resume/cache semantics."""
    from huggingface_hub import snapshot_download

    asset.validate()
    path = snapshot_download(
        repo_id=asset.repo_id,
        repo_type=asset.repo_type,
        revision=asset.revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    target = Path(path)
    if not target.exists():
        raise RuntimeError("Hugging Face reported an absent downloaded snapshot")
    return target
