"""Optional CPU-only source exporter for ten HF datasets.

Nothing is published automatically. Source sha, bounded stream settings and
license-review state are saved for a separate human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .core import canonical_json, dataset_catalog
from .ingest import _atomic_bytes


def group_wikitext(lines: Iterable[dict]):
    article = None
    seq = 0
    for row in lines:
        text = str(row.get("text", ""))
        compact = text.strip()
        if compact.startswith("= ") and compact.endswith(" =") and not compact.startswith("= ="):
            article = hashlib.sha256(compact.encode("utf-8")).hexdigest()[:20]
            seq = 0
            continue
        if article is None or len(compact) < 60:
            continue
        yield {"id": article + ":" + str(seq), "article_id": article, "text": compact}
        seq += 1


def acquire(
    source: str, destination: Path, *, max_raw_rows: int = 20000,
    seed: int = 20261018, source_revision: str | None = None,
) -> dict:
    definitions = {r["id"]: r for r in dataset_catalog()["sources"]}
    if source not in definitions:
        raise ValueError("Unknown public dataset")
    if not 100 <= max_raw_rows <= 200000:
        raise ValueError("CPU source scan limit must be 100..200000")
    if destination.exists():
        raise FileExistsError("Do not overwrite a source export")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError(
            "Install the optional CPU dependencies datasets and huggingface_hub"
        ) from e
    entry = definitions[source]
    if source_revision is not None:
        if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
            raise ValueError("Explicit source revision must be a 40-character lowercase Git SHA")
        revision = HfApi().dataset_info(repo_id=entry["repo"], revision=source_revision).sha
        if revision != source_revision:
            raise ValueError("Requested revision did not match actual source commit")
    else:
        revision = HfApi().dataset_info(repo_id=entry["repo"]).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Could not resolve immutable upstream revision")
    options = {
        "path": entry["repo"],
        "split": entry["split"],
        "revision": revision,
        "streaming": True,
    }
    if entry["config"]:
        options["name"] = entry["config"]
    stream = load_dataset(**options)
    # WikiText is a LINE-LEVEL corpus. Shuffling raw lines first would detach
    # passages from article headers and forge source-group identities.
    if source != "wikitext103":
        stream = stream.shuffle(seed=seed, buffer_size=2048)

    def capped():
        for i, row in enumerate(stream):
            if i >= max_raw_rows:
                break
            if not isinstance(row, dict):
                raise ValueError("Expected source JSON object")
            yield row

    rows = group_wikitext(capped()) if source == "wikitext103" else capped()
    content = b"".join(canonical_json(row) for row in rows)
    if not content:
        raise ValueError("Empty export from pinned upstream data")
    _atomic_bytes(destination, content)
    manifest = {
        "schema": "opensae.source_export.v1",
        "dataset_id": source,
        "repo": entry["repo"],
        "config": entry["config"],
        "split": entry["split"],
        "resolved_revision": revision,
        "shuffle_seed": seed,
        "shuffle_buffer_size": 0 if source == "wikitext103" else 2048,
        "max_raw_rows": max_raw_rows,
        "sampling_scope": (
            "bounded original-order article extraction; deterministic group-ranked selection"
            if source == "wikitext103"
            else "bounded shuffled stream; not uniform across whole source"
        ),
        "source_export_sha256": hashlib.sha256(content).hexdigest(),
        "license_reviewed": False,
        "human_reviewed": False,
        "auto_publish": False,
    }
    _atomic_bytes(
        destination.with_suffix(destination.suffix + ".manifest.json"), canonical_json(manifest)
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, choices=[s["id"] for s in dataset_catalog()["sources"]]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-raw-rows", type=int, default=20000)
    parser.add_argument("--source-revision", help="Optional exact 40-hex pinned dataset commit")
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.source, args.output, max_raw_rows=args.max_raw_rows, source_revision=args.source_revision), indent=2))


if __name__ == "__main__":
    main()
