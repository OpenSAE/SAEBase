"""Read-only Neuronpedia explanations for exactly identified SAE resources.

The Model/SAE API owns source matching and a bounded export cache. Capture
owns which measured feature IDs are reported and whether enrichment completed.
No prompt, activation, credential or private model is uploaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET


EXPORT_ROOT = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


_EXPORT_SESSION = ContextVar("saebase_neuronpedia_export_snapshot", default=None)


@contextmanager
def annotation_session():
    """Reuse a verified in-memory export snapshot for one read-only batch."""
    token = _EXPORT_SESSION.set({})
    try:
        yield
    finally:
        _EXPORT_SESSION.reset(token)


def _source(pair) -> dict | None:
    """Use the resource binding produced by SAE loading, including legacy pairs."""
    binding = getattr(pair.identity, "source", None)
    if binding is None:
        from saebase.api.sae_profiles import get_profile
        from saebase.api.sae_loading import resolve_sae_source

        try:
            profile = get_profile(pair.identity.sae_key)
        except (KeyError, ValueError):
            return None
        binding = resolve_sae_source(profile, pair.identity.layer)
    cfg = getattr(pair.sae, "cfg", None)
    metadata = getattr(cfg, "metadata", None)
    declared = getattr(metadata, "neuronpedia_id", None) or getattr(cfg, "neuronpedia_id", None)
    identifier = binding.get("neuronpedia_id") or declared
    if declared and binding.get("neuronpedia_id") and declared != identifier:
        raise ValueError("Loaded SAE and its source disagree on Neuronpedia identity")
    if not identifier:
        return None
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("Invalid Neuronpedia model/source identifier")
    return {
        "id": identifier,
        "repo": binding["repo"],
        "folder": binding["folder"],
        "d_in": pair.identity.d_in,
        "d_sae": pair.identity.d_sae,
        "hook": pair.identity.hook_name,
        **({"catalog_verified": True} if binding.get("neuronpedia_id") else {}),
    }


def _verify_source(source: dict, expected: dict) -> str:
    if not isinstance(source, dict):
        raise ValueError("Neuronpedia source metadata must be an object")
    model, sae = expected["id"].split("/")
    config = source.get("saelensConfig") or {}
    if not isinstance(config, dict):
        raise ValueError("Neuronpedia SAE configuration must be an object")
    if (source.get("modelId"), source.get("id")) != (model, sae):
        raise ValueError("Neuronpedia returned a different model/SAE source")
    if source.get("hfRepoId") != expected["repo"]:
        raise ValueError("Neuronpedia checkpoint repo/folder does not match the loaded SAE")
    source_folder = source.get("hfFolderId")
    exact_folder = source_folder == expected["folder"]
    # Recent exports name the collection directory and omit SAE config. In that
    # schema, an exact trusted catalog binding identifies the checkpoint; the
    # remote collection must still contain its physical folder.
    catalog_collection = (
        expected.get("catalog_verified") is True
        and isinstance(source_folder, str)
        and bool(source_folder)
        and Path(source_folder) in Path(expected["folder"]).parents
        and not config
    )
    if not exact_folder and not catalog_collection:
        raise ValueError("Neuronpedia checkpoint repo/folder does not match the loaded SAE")
    if not config and expected.get("catalog_verified") is True:
        return "official_catalog_binding; remote dimensions unavailable"
    hook = config.get("hook_name") or config.get("hook_point")
    if (
        config.get("hook_name")
        and config.get("hook_point")
        and config["hook_name"] != config["hook_point"]
    ):
        raise ValueError("Neuronpedia hook_name and legacy hook_point disagree")
    if (config.get("d_in"), config.get("d_sae"), hook) != (
        expected["d_in"],
        expected["d_sae"],
        expected["hook"],
    ):
        raise ValueError("Neuronpedia dimensions/hook do not match the loaded SAE")
    return "remote_repo_folder_dimensions_hook"


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _fetch(url: str, *, timeout: float, limit: int) -> bytes:
    with urlopen(url, timeout=timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Neuronpedia download exceeds the byte budget")
    return data


def _download(expected: dict, options: dict) -> dict:
    from neuronpedia.np_explanation import Explanation

    deadline = time.monotonic() + options["total_timeout_seconds"]
    remaining = options["max_download_bytes"]
    expanded = options["max_uncompressed_bytes"]
    files = []

    def fetch(url):
        nonlocal remaining
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError("Neuronpedia export exceeded the total time budget")
        data = _fetch(url, timeout=min(options["timeout_seconds"], budget), limit=remaining)
        remaining -= len(data)
        files.append({"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return data

    prefix = f"v1/{expected['id']}/"
    source_url = f"{EXPORT_ROOT}/{prefix}source.jsonl"
    source = json.loads(fetch(source_url))
    _verify_source(source, expected)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}

    def listing(params):
        doc = ET.fromstring(fetch(EXPORT_ROOT + "?" + urlencode({"list-type": "2", **params})))
        if doc.findtext("s:IsTruncated", namespaces=ns) != "false":
            raise ValueError("Neuronpedia explanation listing is incomplete")
        return doc

    root = listing({"prefix": prefix, "delimiter": "/"})
    folders = [
        node.findtext("s:Prefix", namespaces=ns) for node in root.findall("s:CommonPrefixes", ns)
    ]
    folders = [
        folder
        for folder in folders
        if isinstance(folder, str)
        and re.fullmatch(re.escape(prefix) + r"explanations(?:-[A-Za-z0-9_.-]+)?/", folder)
    ]
    entries = [
        entry
        for folder in sorted(folders)
        for entry in listing({"prefix": folder}).findall("s:Contents", ns)
    ]
    # Official exports also contain generator settings and failed-index lists.
    # They are not explanations; permit only these exact sidecars in known folders.
    sidecars = {
        folder + name
        for folder in folders
        for name in ("config.json", "failed_explanation_indexes.txt")
    }
    ignored = [
        entry.findtext("s:Key", namespaces=ns)
        for entry in entries
        if entry.findtext("s:Key", namespaces=ns) in sidecars
    ]
    entries = [entry for entry in entries if entry.findtext("s:Key", namespaces=ns) not in sidecars]
    keys = [entry.findtext("s:Key", namespaces=ns) for entry in entries]
    if sum(int(entry.findtext("s:Size", namespaces=ns)) for entry in entries) > remaining:
        raise ValueError("Neuronpedia explanation export exceeds the download budget")
    if any(
        not isinstance(key, str)
        or not re.fullmatch(
            re.escape(prefix) + r"explanations(?:-[A-Za-z0-9_.-]+)?/batch-\d+\.jsonl\.gz", key
        )
        for key in keys
    ):
        raise ValueError("Unexpected Neuronpedia export object path")
    model, sae = expected["id"].split("/")
    from concurrent.futures import ThreadPoolExecutor

    sizes = {
        entry.findtext("s:Key", namespaces=ns): int(entry.findtext("s:Size", namespaces=ns))
        for entry in entries
    }

    def fetch_export(key):
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError("Neuronpedia export exceeded the total time budget")
        packed = _fetch(
            f"{EXPORT_ROOT}/{key}",
            timeout=min(options["timeout_seconds"], budget),
            limit=sizes[key],
        )
        if len(packed) != sizes[key]:
            raise ValueError("Neuronpedia export size changed since listing")
        return key, packed

    explanations = {}
    # Total compressed sizes were checked before dispatch; each worker may read
    # only its listed bytes. Parsing and decompression budgets remain sequential.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, packed in pool.map(fetch_export, sorted(keys)):
            remaining -= len(packed)
            files.append(
                {
                    "url": f"{EXPORT_ROOT}/{key}",
                    "sha256": hashlib.sha256(packed).hexdigest(),
                    "bytes": len(packed),
                }
            )
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
                raw = stream.read(expanded + 1)
            if len(raw) > expanded:
                raise ValueError("Neuronpedia export exceeds the decompressed byte budget")
            expanded -= len(raw)
            for line in raw.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Neuronpedia explanation row must be an object")
                if (row.get("modelId"), row.get("layer", row.get("source"))) != (model, sae):
                    raise ValueError("Explanation belongs to a different Neuronpedia model/source")
                explanation = Explanation.from_np_db_json({**row, "source": sae})
                if not 0 <= explanation.index < expected["d_sae"]:
                    raise ValueError("Neuronpedia feature index is outside the loaded SAE")
                if not isinstance(explanation.text, str) or not explanation.text.strip():
                    continue
                value = {
                    "id": row.get("id"),
                    "description": explanation.text,
                    "method": explanation.method,
                    "explainer_model": explanation.explainer_model,
                    "author_id": row.get("authorId"),
                }
                bucket = explanations.setdefault(str(explanation.index), [])
                if value not in bucket:
                    bucket.append(value)
    return {
        "schema": "saebase.neuronpedia_export.v2",
        "neuronpedia_id": expected["id"],
        "source_metadata": source,
        "explanations": explanations,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "export_folders": folders,
        "ignored_export_objects": ignored,
    }


def feature_activation_densities(pair, options: dict) -> dict:
    """Read native nonzero fractions from the exact SAE's public feature exports.

    Uses annotation budgets and a separate verified cache; values describe the
    Neuronpedia corpus, independently of a capture run's selected prompts.
    """
    import math
    from hydra.utils import to_absolute_path
    from saebase.runtime.experiment_results import write_json

    expected = _source(pair)
    if expected is None:
        raise ValueError("No Neuronpedia source for native feature activation densities")
    path = Path(to_absolute_path(options["cache_dir"])) / expected["id"] / "densities.json"

    def validate(payload):
        if payload["neuronpedia_id"] != expected["id"]:
            raise ValueError("Density cache belongs to a different SAE")
        _verify_source(payload["source_metadata"], expected)
        fractions = payload["fractions"]
        if set(fractions) != {str(i) for i in range(expected["d_sae"])}:
            raise ValueError("Neuronpedia density export has incomplete feature coverage")
        for value in fractions.values():
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError("Invalid native nonzero activation fraction")

    if path.is_file():
        document = json.loads(path.read_text())
        payload = document["payload"]
        if _digest(payload) != document["sha256"]:
            raise ValueError("Neuronpedia density cache checksum mismatch")
        validate(payload)
    else:
        if not options["allow_network"]:
            raise ValueError("No verified Neuronpedia density cache; network explicitly disabled")
        deadline = time.monotonic() + options["total_timeout_seconds"]
        remaining = options["max_download_bytes"]
        expanded = options["max_uncompressed_bytes"]
        files = []

        def fetch(url, limit=None):
            nonlocal remaining
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError("Neuronpedia density export exceeded time budget")
            data = _fetch(
                url,
                timeout=min(options["timeout_seconds"], budget),
                limit=min(remaining, limit) if limit is not None else remaining,
            )
            remaining -= len(data)
            files.append(
                {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            )
            return data

        prefix = f"v1/{expected['id']}/features/"
        source = json.loads(fetch(f"{EXPORT_ROOT}/v1/{expected['id']}/source.jsonl"))
        _verify_source(source, expected)
        listing = ET.fromstring(
            fetch(EXPORT_ROOT + "?" + urlencode({"list-type": "2", "prefix": prefix}))
        )
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        if listing.findtext("s:IsTruncated", namespaces=ns) != "false":
            raise ValueError("Neuronpedia density listing is incomplete")
        entries = []
        for entry in listing.findall("s:Contents", ns):
            key = entry.findtext("s:Key", namespaces=ns)
            if key and re.fullmatch(re.escape(prefix) + r"batch-\d+\.jsonl\.gz", key):
                entries.append((key, int(entry.findtext("s:Size", namespaces=ns))))
        if not entries or sum(size for _, size in entries) > remaining:
            raise ValueError("Neuronpedia density export missing or exceeds byte budget")
        fractions = {}
        model, sae = expected["id"].split("/")
        from concurrent.futures import ThreadPoolExecutor

        def fetch_density(entry):
            key, size = entry
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError("Neuronpedia density export exceeded time budget")
            packed = _fetch(
                f"{EXPORT_ROOT}/{key}",
                timeout=min(options["timeout_seconds"], budget),
                limit=size,
            )
            if len(packed) != size:
                raise ValueError("Neuronpedia density export size changed since listing")
            return key, packed

        # Listed sizes bound total transfer before workers start. Parse in order
        # so expanded-byte accounting and feature uniqueness stay deterministic.
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloaded = list(pool.map(fetch_density, sorted(entries)))
        for key, packed in downloaded:
            remaining -= len(packed)
            files.append(
                {
                    "url": f"{EXPORT_ROOT}/{key}",
                    "sha256": hashlib.sha256(packed).hexdigest(),
                    "bytes": len(packed),
                }
            )
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
                raw = stream.read(expanded + 1)
            if len(raw) > expanded:
                raise ValueError("Neuronpedia density export exceeds decompressed byte budget")
            expanded -= len(raw)
            for line in raw.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("modelId"), row.get("layer")) != (model, sae):
                    raise ValueError("Density belongs to a different Neuronpedia SAE")
                index = str(int(row["index"]))
                if index in fractions:
                    raise ValueError("Duplicate feature in density export")
                fractions[index] = row.get("frac_nonzero")
        payload = {
            "neuronpedia_id": expected["id"],
            "source_metadata": source,
            "fractions": fractions,
            "files": files,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        validate(payload)
        write_json(path, {"payload": payload, "sha256": _digest(payload)})
    return {
        "fractions": payload["fractions"],
        "sha256": _digest(payload),
        "provenance": {k: payload[k] for k in ("files", "retrieved_at_utc", "source_metadata")},
    }


def annotate_features(pair, feature_ids: list[int], options: dict) -> tuple[dict, dict[int, dict]]:
    """Return source-provided hypotheses and explicit lookup outcomes for every ID."""
    from hydra.utils import to_absolute_path
    from saebase.runtime.experiment_results import write_json

    ids = sorted(set(feature_ids))
    info = {
        "provider": "neuronpedia",
        "requested": len(ids),
        "explained": 0,
        "interpretation": "Published feature explanations; not causal validation",
    }

    def unavailable(status, reason):
        info.update(status=status, reason=reason)
        return info, {i: {"status": status, "reason": reason, "explanations": []} for i in ids}

    if not options["enabled"]:
        return unavailable("disabled", "Neuronpedia annotation explicitly disabled")
    try:
        expected = _source(pair)
        if expected is None:
            return unavailable(
                "unsupported", "No public Neuronpedia source is registered for the selected SAE"
            )
        info["neuronpedia_id"] = expected["id"]
        info["export_base_url"] = f"{EXPORT_ROOT}/v1/{expected['id']}/"
        path = (
            Path(to_absolute_path(str(Path(options["cache_dir"]).expanduser())))
            / expected["id"]
            / "explanations.json"
        )
        info["cache_path"] = str(path)
        payload = None
        export_digest = None
        session = _EXPORT_SESSION.get()
        session_key = (str(path), _digest(expected))
        if session is not None and session_key in session:
            payload, export_digest = session[session_key]
            info["retrieval"] = "verified_session_snapshot"
        if payload is None and path.is_file():
            try:
                document = json.loads(path.read_text())
                export_digest = _digest(document["payload"])
                if document["sha256"] != export_digest:
                    raise ValueError("Neuronpedia cache checksum mismatch")
                payload = document["payload"]
                if (
                    payload["neuronpedia_id"] != expected["id"]
                    or payload["schema"] != "saebase.neuronpedia_export.v2"
                ):
                    raise ValueError("Neuronpedia cache identity mismatch")
                _verify_source(payload["source_metadata"], expected)
                if not isinstance(payload.get("explanations"), dict) or not isinstance(
                    payload.get("files"), list
                ):
                    raise ValueError("Invalid Neuronpedia explanation cache structure")
                info["retrieval"] = "cache"
            except (ValueError, KeyError, TypeError) as exc:
                payload = None
                info["cache_rejected"] = str(exc)
        if payload is None:
            if not options["allow_network"]:
                return unavailable(
                    "blocked", "No verified Neuronpedia cache; network explicitly disabled"
                )
            info["attempted_network"] = True
            payload = _download(expected, options)
            export_digest = _digest(payload)
            write_json(path, {"payload": payload, "sha256": export_digest})
            info["retrieval"] = "download"
        if session is not None:
            session[session_key] = (payload, export_digest)
        info.update(
            source_validation=_verify_source(payload["source_metadata"], expected),
            provenance={key: copy.deepcopy(payload[key]) for key in ("retrieved_at_utc", "files")},
            export_sha256=export_digest,
            source_metadata=copy.deepcopy(payload["source_metadata"]),
        )
        rows = {}
        for index in ids:
            values = payload["explanations"].get(str(index), [])
            rows[index] = {
                "status": "available" if values else "missing",
                "explanations": [dict(value) for value in values],
                "url": f"https://www.neuronpedia.org/{expected['id']}/{index}",
            }
        info["explained"] = sum(row["status"] == "available" for row in rows.values())
        info["status"] = "complete" if info["explained"] == len(ids) else "partial"
        return info, rows
    except (ImportError, OSError, ValueError, KeyError, TypeError, ET.ParseError, EOFError) as exc:
        # Preserve the measured capture; the Task completion summary exposes missing annotation.
        return unavailable("blocked", f"{type(exc).__name__}: {exc}")
