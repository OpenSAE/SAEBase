"""Shard-friendly, read-only SQLite browsing index for real SAE observations.

Use when the compact JSON index becomes too large. This is a derived,
checksum-verified *browsing* index, not the lossless scientific activation
store. Never rederive SAR/RSS measurements from its top examples.

The immutable database is written from real, checkpoint-identity-matched
measurement rows and remains unavailable to readers until the manifest and
source DB are finalized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .core import (
    InvalidEvidence,
    canonical_json,
    data_root,
    digest,
    has_complete_identity,
    load_sites,
    read_json,
)
from .ingest import _atomic_bytes, _raw_jsonl

SCHEMA = "opensae.feature_sqlite.v1"
DATABASE = "features.sqlite3"
MAX_EXAMPLES = 8

DDL = """
CREATE TABLE features (
    feature_key TEXT PRIMARY KEY NOT NULL,
    site_id TEXT NOT NULL,
    model TEXT NOT NULL,
    sae TEXT NOT NULL,
    layer INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    observations INTEGER NOT NULL DEFAULT 0,
    unique_prompt_groups INTEGER NOT NULL DEFAULT 0,
    max_activation REAL NOT NULL,
    label TEXT,
    label_status TEXT NOT NULL DEFAULT 'unlabeled',
    source_type TEXT NOT NULL CHECK (source_type = 'real_checkpoint')
);
CREATE INDEX feature_model_id ON features(model, layer, feature_id);
CREATE INDEX feature_site_id ON features(site_id, feature_id);
CREATE TABLE seen_groups (
    feature_key TEXT NOT NULL REFERENCES features(feature_key),
    prompt_group_id TEXT NOT NULL,
    PRIMARY KEY (feature_key, prompt_group_id)
) WITHOUT ROWID;
CREATE TABLE examples (
    feature_key TEXT NOT NULL REFERENCES features(feature_key),
    sample_id TEXT NOT NULL,
    token_position INTEGER NOT NULL,
    activation REAL NOT NULL,
    prompt_group_id TEXT NOT NULL,
    token_text TEXT NOT NULL,
    text_excerpt TEXT NOT NULL,
    PRIMARY KEY (feature_key, sample_id, token_position)
) WITHOUT ROWID;
CREATE INDEX example_rank ON examples(feature_key, activation DESC);
"""


def _site_lookup(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {s["site_id"]: s for s in load_sites(repo_root)}


def _validate_observation(row: dict[str, Any], sites: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str, int, int, float, str, str]:
    site = sites.get(row.get("site_id"))
    if (
        site is None
        or not has_complete_identity(site)
        or row.get("model_sha256") != site["model_sha256"]
        or row.get("sae_sha256") != site["sae_sha256"]
    ):
        raise ValueError("Unverified checkpoint/site or missing original weight SHA-256")
    if row.get("source_type") != "real_checkpoint":
        raise ValueError("Synthetic or unspecified observation rejected")
    feature, position = row.get("feature_id"), row.get("token_position")
    if type(feature) is not int or feature < 0 or type(position) is not int or position < 0:
        raise ValueError("Expected nonnegative integer SAE feature and token indices")
    activation = row.get("activation")
    if type(activation) not in (int, float) or not math.isfinite(activation):
        raise ValueError("Nonfinite or invalid activation")
    group, sample = row.get("prompt_group_id"), row.get("sample_id")
    if not isinstance(group, str) or not group.strip() or not isinstance(sample, str) or not sample.strip():
        raise ValueError("Missing original prompt-group or sample identity")
    key = site["site_id"] + ":F" + str(feature)
    return site, key, feature, position, float(activation), group, sample


def create_feature_store(
    observations: Iterable[dict[str, Any]],
    output: Path,
    *,
    repo_root: Path | None = None,
    max_observations: int = 100_000_000,
) -> dict[str, Any]:
    """Ingest real observations with bounded memory and top-8 public excerpts."""
    if (output / "manifest.json").exists() or (output / DATABASE).exists():
        raise FileExistsError("An immutable feature index already exists")
    if not 1 <= max_observations <= 100_000_000:
        raise ValueError("Invalid ingestion budget")
    sites = _site_lookup(repo_root)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (DATABASE + ".partial")
    if temporary.exists():
        raise FileExistsError("Unfinished index present; inspect before deleting it")
    count = 0
    public_rows = 0
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(DDL)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 1000")
        connection.execute("BEGIN IMMEDIATE")
        for row in observations:
            if count >= max_observations:
                raise ValueError("Observation budget exceeded")
            site, key, feature_id, position, strength, group, sample = _validate_observation(row, sites)
            connection.execute(
                """INSERT OR IGNORE INTO features
                (feature_key,site_id,model,sae,layer,feature_id,max_activation,source_type)
                VALUES (?,?,?,?,?,?,?,'real_checkpoint')""",
                (key, site["site_id"], site["model"], site["sae"], site["layer"], feature_id, strength),
            )
            connection.execute(
                """UPDATE features SET observations=observations+1,
                max_activation=MAX(max_activation, ?) WHERE feature_key=?""",
                (strength, key),
            )
            if connection.execute(
                "INSERT OR IGNORE INTO seen_groups(feature_key,prompt_group_id) VALUES (?,?)",
                (key,group),
            ).rowcount:
                connection.execute(
                    "UPDATE features SET unique_prompt_groups=unique_prompt_groups+1 WHERE feature_key=?",
                    (key,),
                )
            if row.get("public_text_reviewed") is True:
                token = row.get("token_text", "")
                excerpt = row.get("text_excerpt", "")
                if not isinstance(token, str) or not isinstance(excerpt, str):
                    raise ValueError("Reviewed excerpts must be text")
                # Avoid writing unbounded user-controlled source content to DB.
                connection.execute(
                    """INSERT OR REPLACE INTO examples
                    (feature_key,sample_id,token_position,activation,prompt_group_id,token_text,text_excerpt)
                    VALUES (?,?,?,?,?,?,?)""",
                    (key,sample,position,strength,group,token[:128],excerpt[:700]),
                )
                candidates = connection.execute(
                    """SELECT sample_id,token_position FROM examples
                    WHERE feature_key=? ORDER BY activation DESC,sample_id,token_position
                    LIMIT ?""",
                    (key,MAX_EXAMPLES),
                ).fetchall()
                if len(candidates) == MAX_EXAMPLES:
                    # The insert may have grown the list to nine. Keep the
                    # strongest entries in a deterministic tie order.
                    keys = {(a,b) for a,b in candidates}
                    for a,b in connection.execute(
                        "SELECT sample_id,token_position FROM examples WHERE feature_key=?",
                        (key,),
                    ).fetchall():
                        if (a,b) not in keys:
                            connection.execute(
                                "DELETE FROM examples WHERE feature_key=? AND sample_id=? AND token_position=?",
                                (key,a,b),
                            )
                public_rows += 1
            count += 1
            if count % 10000 == 0:
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
        connection.commit()
        if count == 0:
            raise ValueError("Feature store needs genuine observations")
        # Compact the small publish-time derived database. No WAL sidecars.
        connection.execute("VACUUM")
        feature_count = connection.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        grouped = connection.execute("SELECT COUNT(*) FROM seen_groups").fetchone()[0]
        displayed = connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
    except Exception:
        connection.rollback()
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    os.replace(temporary, output / DATABASE)
    manifest = {
        "schema": SCHEMA,
        "files": {DATABASE: digest(output / DATABASE)},
        "feature_count": feature_count,
        "observation_count": count,
        "unique_group_feature_pairs": grouped,
        "reviewed_text_observations": public_rows,
        "public_examples_stored": displayed,
        "browse_only": True,
        "source_type": "real_checkpoint",
    }
    _atomic_bytes(output / "manifest.json", canonical_json(manifest))
    return manifest


def _verified_store(root: Path | None = None) -> Path | None:
    folder = (root or data_root()) / "features"
    manifest_path, db_path = folder / "manifest.json", folder / DATABASE
    if not db_path.exists() and not manifest_path.exists():
        return None
    if not manifest_path.is_file() or not db_path.is_file():
        raise InvalidEvidence("Incomplete feature SQLite release")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        # Allow compact v1 JSON backend to handle its own manifest.
        if manifest.get("schema") == "opensae.feature_index.v1":
            return None
        raise InvalidEvidence("Unknown feature index schema")
    if digest(db_path) != manifest.get("files", {}).get(DATABASE):
        raise InvalidEvidence("Feature SQLite checksum mismatch")
    return db_path


def query_feature_store(
    *,
    root: Path | None = None,
    query: str = "",
    model: str = "",
    site_id: str = "",
    limit: int = 36,
    offset: int = 0,
) -> dict[str, Any] | None:
    """Pagination with parameterized filters and immutable database."""
    path = _verified_store(root)
    if path is None:
        return None
    if not 1 <= limit <= 100 or not 0 <= offset <= 10000000:
        raise ValueError("Invalid pagination")
    clauses: list[str] = []
    params: list[Any] = []
    if model:
        clauses.append("model LIKE ? ESCAPE '!'")
        params.append("%" + _escape_like(model[:120]) + "%")
    if site_id:
        clauses.append("site_id=?")
        params.append(site_id[:300])
    if query:
        clauses.append("(feature_key LIKE ? ESCAPE '!' OR COALESCE(label,'') LIKE ? ESCAPE '!')")
        params.extend(["%" + _escape_like(query[:120]) + "%"] * 2)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        total = conn.execute("SELECT COUNT(*) FROM features" + where, params).fetchone()[0]
        rows = [
            dict(r) for r in conn.execute(
                "SELECT * FROM features" + where +
                " ORDER BY model,layer,feature_id LIMIT ? OFFSET ?",
                [*params,limit,offset],
            )
        ]
        for row in rows:
            examples = conn.execute(
                """SELECT sample_id,token_position,activation,prompt_group_id,
                token_text,text_excerpt FROM examples WHERE feature_key=?
                ORDER BY activation DESC,sample_id,token_position LIMIT ?""",
                (row["feature_key"], MAX_EXAMPLES),
            ).fetchall()
            row["examples"] = [dict(r) for r in examples]
    finally:
        conn.close()
    return {"items":rows, "total":total, "limit":limit, "offset":offset}


def _escape_like(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def feature_store_count(root: Path | None = None) -> int | None:
    path = _verified_store(root)
    if path is None:
        return None
    meta = read_json(path.parent / "manifest.json")
    return int(meta["feature_count"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-observations", type=int, default=100_000_000)
    args = parser.parse_args(argv)
    results = create_feature_store(
        _raw_jsonl(args.input),
        output=args.output or data_root() / "features",
        max_observations=args.max_observations,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
