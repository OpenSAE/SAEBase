"""Read-only exploration of source-bound SAE feature and token records.

CPU-only, no remote inference. The published trace collection is small and
checksummed; the independently maintained feature SQLite index handles large
dictionaries. Missing observations are unknown, never assumed to be zero.
"""

from __future__ import annotations

import hashlib
import html
import math
from pathlib import Path
from typing import Any, Iterable

from . import core
from .feature_store import query_feature_store
from .ingest import _atomic_bytes

SCHEMA = "opensae.explore.release.v1"
FILE = "records.json"
MAX_TRACES = 32
MAX_TOKENS = 128
MAX_ACTIVE = 256
MAX_BYTES = 8 * 1024 * 1024
MAX_LABELS = 1024


def _sha(value: str, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _sites(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    return {site["site_id"]: site for site in core.load_sites(repo_root)}


def _source(row: dict[str, Any], sites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    site = sites.get(row.get("site_id"))
    if (
        site is None
        or not core.has_complete_identity(site)
        or row.get("model_sha256") != site["model_sha256"]
        or row.get("sae_sha256") != site["sae_sha256"]
        or row.get("source_type") != "real_checkpoint"
    ):
        raise ValueError("Trace or explanation lacks an exact verified source site")
    return site


def _valid_trace(row: dict[str, Any], sites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    site = _source(row, sites)
    if row.get("public_text_reviewed") is not True or row.get("rights_reviewed") is not True:
        raise PermissionError("Only reviewed public text and observations may be released")
    prompt = row.get("prompt_text")
    if not isinstance(prompt, str) or not prompt or len(prompt) > 16000:
        raise ValueError("Invalid prompt text")
    if row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise ValueError("Prompt hash mismatch")
    if not _sha(row.get("tokenizer_sha256")):
        raise ValueError("Exact tokenizer digest required")
    for field in ("sample_id", "prompt_group_id", "model_processing"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError("Missing trace identity: " + field)
    if row.get("split") not in {"discovery", "evaluation"}:
        raise ValueError("Explicit discovery/evaluation split required")
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_TOKENS:
        raise ValueError("Trace must contain 1–128 ordered tokens")
    if row.get("complete_sparse_codes") is not True:
        raise ValueError("Only complete nonzero code coverage permits recorded inference")
    for position, token in enumerate(tokens):
        if not isinstance(token, dict) or type(token.get("position")) is not int:
            raise ValueError("Malformed token row")
        if token["position"] != position or type(token.get("token_id")) is not int:
            raise ValueError("Token IDs/positions must be complete and aligned")
        if not isinstance(token.get("text"), str) or len(token["text"]) > 128:
            raise ValueError("Invalid token text")
        features = token.get("features")
        if not isinstance(features, list) or len(features) > MAX_ACTIVE:
            raise ValueError("Unbounded or missing sparse codes")
        seen: set[int] = set()
        for feature in features:
            if not isinstance(feature, dict):
                raise ValueError("Invalid sparse feature")
            fid, strength = feature.get("feature_id"), feature.get("activation")
            if type(fid) is not int or fid < 0 or fid in seen:
                raise ValueError("Invalid or repeated sparse feature ID")
            if type(strength) not in (int, float) or not math.isfinite(strength) or strength == 0:
                raise ValueError("Sparse feature activation must be nonzero and finite")
            seen.add(fid)
    # Publish an allowlisted record only. Source-worker diagnostics, raw text
    # fields or accidental tokens must not silently leak via **row.
    allowed = (
        "site_id", "model_sha256", "sae_sha256", "source_type", "sample_id",
        "prompt_group_id", "split", "prompt_text", "prompt_sha256",
        "tokenizer_sha256", "model_processing", "complete_sparse_codes",
        "public_text_reviewed", "rights_reviewed",
    )
    copy = {key: row[key] for key in allowed}
    copy["tokens"] = [
        {
            "position": token["position"], "token_id": token["token_id"],
            "text": token["text"], "features": [
                {"feature_id": feature["feature_id"], "activation": float(feature["activation"])}
                for feature in token["features"]
            ],
        }
        for token in tokens
    ]
    copy["site"] = {key: site[key] for key in ("model", "sae", "layer", "hook")}
    copy["trace_id"] = hashlib.sha256(core.canonical_json(copy)).hexdigest()
    return copy


def _valid_explanation(row: dict[str, Any], sites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    _source(row, sites)
    fid = row.get("feature_id")
    if type(fid) is not int or fid < 0:
        raise ValueError("Invalid explanation feature ID")
    if row.get("approved") is not True:
        raise PermissionError("Unreviewed explanation cannot be made public")
    text = row.get("description")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 900:
        raise ValueError("Invalid explanation text")
    if row.get("origin") not in {"human", "model_candidate"}:
        raise ValueError("Unrecognized explanation origin")
    for field in ("method", "author_or_model"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError("Missing explanation provenance")
    if row.get("status") not in {"hypothesis", "heldout_tested"}:
        raise ValueError("Interpretation must be labeled a hypothesis or tested candidate")
    discovery = row.get("discovery_group_ids", [])
    evaluation = row.get("evaluation_group_ids", [])
    if (
        not isinstance(discovery, list)
        or not isinstance(evaluation, list)
        or any(not isinstance(x, str) or not x for x in discovery + evaluation)
        or len(set(discovery)) != len(discovery)
        or len(set(evaluation)) != len(evaluation)
        or set(discovery) & set(evaluation)
    ):
        raise ValueError("Discovery/evaluation groups must be distinct")
    score = row.get("heldout_score")
    if row["status"] == "heldout_tested":
        if (
            not discovery
            or not evaluation
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Held-out score requires exact disjoint groups and finite value")
    elif score is not None:
        raise ValueError("Untested interpretation cannot have a numerical score")
    allowed = (
        "site_id", "model_sha256", "sae_sha256", "source_type",
        "feature_id", "origin", "description", "author_or_model", "method",
        "status", "approved", "discovery_group_ids", "evaluation_group_ids",
        "heldout_score",
    )
    result = {key: row[key] for key in allowed if key in row}
    result["discovery_group_ids"] = discovery
    result["evaluation_group_ids"] = evaluation
    result["heldout_score"] = score
    result["feature_key"] = f"{row['site_id']}:F{fid}"
    return result


def publish_exploration(
    traces: Iterable[dict[str, Any]],
    explanations: Iterable[dict[str, Any]],
    *,
    output: Path,
    repo_root: Path | None = None,
    public_consent: bool = False,
) -> dict[str, Any]:
    """Publish *reviewed* real observations, never edit an existing release."""
    if not public_consent:
        raise PermissionError("Explicit operator approval required to publish traces")
    if (output / "manifest.json").exists() or (output / FILE).exists():
        raise FileExistsError("Exploration release is immutable")
    sites = _sites(repo_root)
    records = [_valid_trace(row, sites) for row in traces]
    labels = [_valid_explanation(row, sites) for row in explanations]
    if len(records) > MAX_TRACES or len(labels) > MAX_LABELS or not (records or labels):
        raise ValueError("Exploration release size out of bounds")
    if len({row["trace_id"] for row in records}) != len(records):
        raise ValueError("Repeated trace")
    if len({(row["site_id"], row["sample_id"]) for row in records}) != len(records):
        raise ValueError("Duplicate site/sample token trace")
    # Labels are hypotheses about *observed* SAE features, not arbitrary
    # invented semantic IDs. Accept an exact reviewed source index or a full
    # verified, approved per-token trace containing that feature.
    observed = {
        (trace["site_id"], code["feature_id"])
        for trace in records for token in trace["tokens"] for code in token["features"]
    }
    from .feature_store import feature_by_id, feature_store_count
    has_sqlite = feature_store_count(output.parent) is not None
    indexed = None if has_sqlite else core.feature_index(output.parent)
    for label in labels:
        key = (label["site_id"], label["feature_id"])
        if key in observed:
            continue
        if has_sqlite:
            exists = feature_by_id(*key, root=output.parent) is not None
        else:
            exists = any(
                item.get("site_id") == key[0] and item.get("feature_id") == key[1]
                for item in indexed
            )
        if not exists:
            raise ValueError("Cannot publish an explanation for an unobserved feature")
    document = {
        "schema": SCHEMA,
        "traces": sorted(records, key=lambda item: (item["site_id"], item["sample_id"])),
        "explanations": sorted(labels, key=lambda item: (item["feature_key"], item["description"])),
    }
    content = core.canonical_json(document)
    if len(content) > MAX_BYTES:
        raise ValueError("Released exploration payload exceeds 8 MiB")
    file_path = output / FILE
    _atomic_bytes(file_path, content)
    manifest = {
        "schema": SCHEMA,
        "files": {FILE: hashlib.sha256(content).hexdigest()},
        "trace_count": len(records),
        "explanation_count": len(labels),
        "model_computed": True if records else False,
        "publication": "reviewed_real_only",
    }
    _atomic_bytes(output / "manifest.json", core.canonical_json(manifest))
    return manifest


def published_exploration(root: Path | None = None) -> dict[str, Any]:
    """Fail closed on changed or unrecognized source; absence is an empty release."""
    folder = (root or core.data_root()) / "explore"
    if not (folder / "manifest.json").exists() and not (folder / FILE).exists():
        return {"schema": SCHEMA, "traces": [], "explanations": []}
    manifest = folder / "manifest.json"
    payload = folder / FILE
    if not manifest.is_file() or not payload.is_file() or payload.stat().st_size > MAX_BYTES:
        raise core.InvalidEvidence("Incomplete or oversized exploration release")
    meta = core.read_json(manifest)
    if meta.get("schema") != SCHEMA or core.digest(payload) != meta.get("files", {}).get(FILE):
        raise core.InvalidEvidence("Exploration source integrity mismatch")
    body = core.read_json(payload)
    if body.get("schema") != SCHEMA:
        raise core.InvalidEvidence("Exploration schema mismatch")
    traces, labels = body.get("traces"), body.get("explanations")
    if (
        not isinstance(traces, list)
        or not isinstance(labels, list)
        or len(traces) != meta.get("trace_count")
        or len(labels) != meta.get("explanation_count")
        or len(traces) > MAX_TRACES
        or len(labels) > MAX_LABELS
    ):
        raise core.InvalidEvidence("Exploration counts invalid")
    sites = _sites()
    try:
        if any(_valid_trace(row, sites)["trace_id"] != row.get("trace_id") for row in traces):
            raise ValueError("Changed trace identity")
        for label in labels:
            _valid_explanation(label, sites)
    except (ValueError, TypeError, KeyError, PermissionError) as exc:
        raise core.InvalidEvidence("Exploration identity, provenance or consent invalid") from exc
    return body


def feature_search(
    *,
    query: str = "",
    model: str = "",
    site_id: str = "",
    mode: str = "feature",
    limit: int = 36,
    offset: int = 0,
    root: Path | None = None,
) -> dict[str, Any]:
    """Search either recorded IDs or published explanation text, not live inference."""
    if mode not in {"feature", "explanation"}:
        raise ValueError("Unknown feature-search mode")
    if not 1 <= limit <= 100 or not 0 <= offset <= 10_000_000:
        raise ValueError("Invalid pagination")
    if mode == "explanation":
        labels = published_exploration(root)["explanations"]
        query_fold = query.casefold()
        hits = [
            x
            for x in labels
            if query_fold in x["description"].casefold()
            and (not site_id or x["site_id"] == site_id)
        ]
        sites = _sites()
        if model:
            hits = [x for x in hits if model.casefold() in sites[x["site_id"]]["model"].casefold()]
        by_key = {x["feature_key"]: x for x in hits}
        keys = sorted(by_key)
        return {
            "total": len(keys),
            "items": [
                {**by_key[key], "site": sites[by_key[key]["site_id"]]}
                for key in keys[offset : offset + limit]
            ],
            "mode": "explanation",
            "source": "approved_descriptions",
        }
    indexed = query_feature_store(
        root=root,
        query=query,
        model=model,
        site_id=site_id,
        limit=limit,
        offset=offset,
    )
    if indexed is not None:
        return {**indexed, "mode": "feature", "source": "verified_sqlite"}
    rows = core.feature_index(root)
    lookup = query.casefold()
    rows = [
        x
        for x in rows
        if (not model or model.casefold() in x["model"].casefold())
        and (not site_id or x["site_id"] == site_id)
        and (not query or lookup in (x["feature_key"] + " " + str(x.get("label") or "")).casefold())
    ]
    rows.sort(key=lambda x: (x["model"], x["layer"], x["feature_id"]))
    return {
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "items": rows[offset : offset + limit],
        "mode": "feature",
        "source": "verified_compact_index",
    }


def feature_details(
    site_id: str, feature_id: int, *, root: Path | None = None
) -> dict[str, Any] | None:
    if type(feature_id) is not int or feature_id < 0:
        return None
    if site_id not in _sites():
        return None
    from .feature_store import feature_by_id, feature_store_count

    has_sqlite = feature_store_count(root) is not None
    result = feature_by_id(site_id, feature_id, root=root) if has_sqlite else None
    if result is None and not has_sqlite:
        results = core.feature_index(root)
        candidates = sorted(
            (row for row in results if row.get("site_id") == site_id),
            key=lambda row: row["feature_id"],
        )
        for index, row in enumerate(candidates):
            if row.get("feature_id") == feature_id:
                result = {
                    **row,
                    "previous_feature_id": (candidates[index - 1]["feature_id"] if index else None),
                    "next_feature_id": (
                        candidates[index + 1]["feature_id"] if index + 1 < len(candidates) else None
                    ),
                }
                break
    if result is None:
        return None
    site = _sites()[site_id]
    labels = [
        x
        for x in published_exploration(root)["explanations"]
        if x["site_id"] == site_id and x["feature_id"] == feature_id
    ]
    examples = result.get("examples") or []
    # Distribution is explicitly *reviewed examples only*, not density across
    # the source corpus; sparse index has no all-token denominator.
    positives = [float(x["activation"]) for x in examples if x["activation"] > 0]
    negative = [float(x["activation"]) for x in examples if x["activation"] < 0]
    return {
        **result,
        "site": {
            k: site[k]
            for k in (
                "site_id",
                "model",
                "sae",
                "layer",
                "hook",
                "model_sha256",
                "sae_sha256",
                "evidence_url",
            )
        },
        "explanations": labels,
        "distribution": {
            "scope": "top_reviewed_examples_only",
            "positive_activations": positives,
            "negative_activations": negative,
            "population_density": None,
        },
        "canonical_url_fragment": "explore?site=" + site_id + "&feature=" + str(feature_id),
        "supported": {"logit_attribution": False, "live_inference": False},
    }


def recorded_traces(*, site_id: str = "", root: Path | None = None) -> list[dict[str, Any]]:
    return [
        {
            "trace_id": x["trace_id"],
            "site_id": x["site_id"],
            "sample_id": x["sample_id"],
            "prompt_group_id": x["prompt_group_id"],
            "token_count": len(x["tokens"]),
            "split": x["split"],
            "prompt_text": x["prompt_text"],
        }
        for x in published_exploration(root)["traces"]
        if not site_id or x["site_id"] == site_id
    ]


def recorded_trace(trace_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    if not _sha(trace_id):
        return None
    return next(
        (x for x in published_exploration(root)["traces"] if x["trace_id"] == trace_id),
        None,
    )


def recorded_inference(
    prompt: str, *, site_id: str = "", root: Path | None = None
) -> list[dict[str, Any]]:
    """Retrieve matching frozen actual input, never imply fresh GPU inference."""
    if not isinstance(prompt, str) or not prompt or len(prompt) > 16000:
        raise ValueError("Input text required")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    return [
        {
            "trace_id": t["trace_id"],
            "site_id": t["site_id"],
            "max_features": strongest_features(t),
            "mode": "recorded_only",
        }
        for t in published_exploration(root)["traces"]
        if t["prompt_sha256"] == digest and (not site_id or t["site_id"] == site_id)
    ]


def strongest_features(
    trace: dict[str, Any], *, token_position: int | None = None, top_k: int = 20
) -> list[dict[str, Any]]:
    """Max activation per feature on selected token(s) of one COMPLETE trace."""
    if not 1 <= top_k <= 100:
        raise ValueError("Invalid top-k request")
    tokens = trace["tokens"]
    if token_position is not None and not 0 <= token_position < len(tokens):
        raise IndexError("Token position out of range")
    collection = tokens if token_position is None else [tokens[token_position]]
    best: dict[int, tuple[float, int]] = {}
    for token in collection:
        for feature in token["features"]:
            fid, value = feature["feature_id"], feature["activation"]
            if fid not in best or abs(value) > abs(best[fid][0]):
                best[fid] = (float(value), token["position"])
    return [
        {
            "feature_id": fid,
            "activation": value,
            "position": pos,
            "feature_key": trace["site_id"] + ":F" + str(fid),
        }
        for fid, (value, pos) in sorted(best.items(), key=lambda x: (-abs(x[1][0]), x[0]))[:top_k]
    ]


def correlations(
    site_id: str, feature_id: int, *, root: Path | None = None, limit: int = 10
) -> dict[str, Any]:
    """Pearson correlation of per-group maximum SAE codes, complete sources only.

    Empty or unobserved groups are zeros ONLY when the trace explicitly covers
    all sparse nonzero codes for every token, not when recording top-k snippets.
    Resample units are independent prompt groups rather than individual tokens.
    """
    if not 1 <= limit <= 40:
        raise ValueError("Invalid correlation limit")
    traces = [t for t in published_exploration(root)["traces"] if t["site_id"] == site_id]
    groups: dict[str, dict[int, float]] = {}
    for trace in traces:
        by_id = groups.setdefault(trace["prompt_group_id"], {})
        for token in trace["tokens"]:
            for feature in token["features"]:
                fid, v = feature["feature_id"], float(feature["activation"])
                if fid not in by_id or abs(v) > abs(by_id[fid]):
                    by_id[fid] = v
    if len(groups) < 3:
        return {
            "items": [],
            "n_groups": len(groups),
            "note": "Need >=3 independent published prompt groups",
        }
    import numpy as np

    array = list(groups.values())
    v = np.asarray([g.get(feature_id, 0.0) for g in array], dtype=float)
    if float(np.std(v)) == 0:
        return {"items": [], "n_groups": len(groups), "note": "Selected feature has zero variance"}
    ids = sorted({fid for g in array for fid in g} - {feature_id})
    found = []
    for fid in ids:
        w = np.asarray([g.get(fid, 0.0) for g in array], dtype=float)
        if np.std(w) == 0:
            continue
        rho = float(np.corrcoef(v, w)[0, 1])
        if math.isfinite(rho):
            found.append({"feature_id": fid, "pearson_r": rho})
    found.sort(key=lambda x: (-abs(x["pearson_r"]), x["feature_id"]))
    return {
        "items": found[:limit],
        "n_groups": len(groups),
        "scope": "complete_sparse_codes_per_independent_prompt_group",
        "note": "Correlation is not causal identity; small-group estimates are unstable",
    }


def feature_html(card: dict[str, Any]) -> str:
    """Self-contained printable HTML; only reviewed public content is inserted."""
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)
    examples = "".join(
        "<li><strong>"
        + esc(e["token_text"])
        + "</strong> · "
        + esc(e["activation"])
        + "<p>"
        + esc(e["text_excerpt"])
        + "</p></li>"
        for e in card.get("examples", [])
    )
    labels = "".join(
        "<li>"
        + esc(item["description"])
        + " <small>("
        + esc(item["origin"])
        + "; "
        + esc(item["status"])
        + "; "
        + esc(item["author_or_model"])
        + ")</small></li>"
        for item in card.get("explanations", [])
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>OpenSAE Feature " + esc(card["feature_id"]) + "</title>"
        "<style>body{font:16px system-ui;max-width:800px;margin:3rem auto;padding:1rem}"
        "li{margin:1rem 0}small{color:#666}p{white-space:pre-wrap}</style>"
        "<h1>Feature " + esc(card["feature_id"]) + "</h1>"
        "<p>Model "
        + esc(card["site"]["model"])
        + " · SAE "
        + esc(card["site"]["sae"])
        + " · "
        + esc(card["site"]["hook"])
        + "</p>"
        "<p>Recorded activity: "
        + esc(card["observations"])
        + " occurrences from "
        + esc(card["unique_prompt_groups"])
        + " groups.</p>"
        "<p>These are recorded observations, not live inference or an explanation guarantee.</p>"
        "<h2>Published interpretations</h2><ul>" + labels + "</ul>"
        "<h2>Reviewed activation examples</h2><ul>" + examples + "</ul>"
        "<footer>OpenSAE SAEBase · verified-site evidence reference: "
        + esc(card["site"]["evidence_url"])
        + "</footer></html>"
    )
