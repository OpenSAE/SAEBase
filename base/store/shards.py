"""Crash-consistent, bounded-shard arrays and optional lossless CSR SAE codes.

The manifest is the commit point. Shards and their sidecars are published first;
only references present in manifest are readable. Uncommitted orphan files are
never reused, and strict reads verify their SHA-256 byte digests before returning
rows. Writer locks cover the manifest for one local filesystem writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Mapping
import hashlib
import json
import os
import re
import tempfile

import numpy as np
import torch

from .identity import (ActivationIdentity, CodeIdentity, SampleRecord,
                       fingerprint, resolve_positions, stable_json)

SCHEMA = "saebase.sharded_store.v1"
DEFAULT_SHARD_BYTES = 128 * 1024 * 1024
_ALLOWED_LAYOUTS = {"dense", "csr"}


class CacheIntegrityError(RuntimeError):
    """A committed shard, manifest, or sample was modified or lost."""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json(path: Path, obj: Any) -> None:
    _atomic_bytes(path, (stable_json(obj) + "\n").encode("utf-8"))


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    """Stream raw matrix data to .npy without pickle or path-suffix surprises."""
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise CacheIntegrityError(f"unclaimed partial array at {temp}")
    try:
        mm = np.lib.format.open_memmap(temp, mode="w+", dtype=array.dtype,
                                       shape=array.shape)
        mm[:] = array
        mm.flush()
        del mm
        with temp.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _to_storage(array: np.ndarray, dtype: str, *, max_quantization_error: float | None) -> tuple[np.ndarray, float]:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().float().numpy()
    original = np.asarray(array)
    if original.ndim != 2 or original.size == 0 or original.dtype.kind not in "fiu":
        raise ValueError("matrix must have shape [positions, width] and real numeric values")
    if not bool(np.isfinite(original).all()):
        raise ValueError("cache entries must contain only finite values")
    original = original.astype(np.float32, copy=False)
    if dtype == "float32":
        result, error = np.asarray(original, dtype=np.float32), 0.0
    elif dtype == "float16":
        result = original.astype(np.float16)
        error = float(np.max(np.abs(original - result.astype(np.float32))))
    elif dtype == "bfloat16":
        encoded = torch.from_numpy(np.array(original, copy=True)).to(torch.bfloat16)
        result = encoded.view(torch.int16).numpy().view(np.uint16)
        decoded = encoded.float().numpy()
        error = float(np.max(np.abs(original - decoded)))
    else:
        raise ValueError("unknown dtype")
    if not bool(np.isfinite(error)) or (max_quantization_error is not None and error > max_quantization_error):
        raise ValueError(f"storage conversion error {error} exceeds {max_quantization_error}")
    return np.ascontiguousarray(result), error


def _from_storage(data: np.ndarray, dtype: str) -> np.ndarray:
    if dtype != "bfloat16":
        return data
    # torch works with uint16 BF16 payloads only after a signed integer view.
    raw = np.array(data, dtype=np.uint16, copy=True).view(np.int16)
    return torch.from_numpy(raw).view(torch.bfloat16).float().numpy()


def _safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
        raise CacheIntegrityError(f"unsafe file name in shard manifest: {name!r}")
    return root / "shards" / name


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("expected an object")
        return obj
    except (OSError, ValueError, UnicodeError) as exc:
        raise CacheIntegrityError(f"invalid manifest/sidecar {path}: {exc}") from exc


@dataclass(frozen=True)
class StoredSample:
    record: SampleRecord
    positions: tuple[int, ...]
    values: np.ndarray  # dense even when codes are stored in lossless sparse CSR


class ShardedReader:
    """Read data through a manifest snapshot; expose mmap-backed dense rows."""

    def __init__(self, path: str | Path, *, verify_shards: bool = True,
                 expected_identity: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.manifest = _load_json(self.path / "manifest.json")
        m = self.manifest
        if m.get("schema") != SCHEMA:
            raise CacheIntegrityError("unsupported store schema")
        if m.get("layout") not in _ALLOWED_LAYOUTS or type(m.get("width")) is not int or m["width"] < 1:
            raise CacheIntegrityError("invalid store layout or width")
        if m.get("dtype") not in {"float32", "float16", "bfloat16"}:
            raise CacheIntegrityError("invalid store dtype")
        if m.get("kind") not in {"activations", "codes"}:
            raise CacheIntegrityError("invalid store kind")
        if fingerprint(m.get("identity")) != m.get("identity_key"):
            raise CacheIntegrityError("store identity hash differs from manifest identity")
        if expected_identity is not None and stable_json(expected_identity) != stable_json(m["identity"]):
            raise CacheIntegrityError("requested model/SAE identity does not match existing cache")
        self.width = m["width"]
        self.layout = m["layout"]
        self.dtype = m["dtype"]
        self.kind = m["kind"]
        self._index: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._ordered_ids: list[str] = []
        self._verified = set()
        self._mmap: OrderedDict[int, Any] = OrderedDict()
        for shard in m["shards"]:
            self._check_sidecar(shard)
            if verify_shards:
                self._check_shard(shard)
            meta = _load_json(_safe_path(self.path, shard["sidecar"]))
            for entry in meta["entries"]:
                record = SampleRecord.from_dict(entry["record"])
                if entry.get("record_sha256") != record.key:
                    raise CacheIntegrityError("sample metadata digest differs")
                if record.sample_id in self._index:
                    raise CacheIntegrityError(f"duplicate committed sample {record.sample_id}")
                positions = tuple(entry["positions"])
                if positions != resolve_positions(record) or (entry["end"] - entry["start"]) != len(positions):
                    raise CacheIntegrityError("sample positions or length differ")
                if not 0 <= entry["start"] < entry["end"] <= shard["rows"]:
                    raise CacheIntegrityError("sample row slice outside shard")
                self._index[record.sample_id] = (shard, entry)
                self._ordered_ids.append(record.sample_id)

    def _check_sidecar(self, shard: Mapping[str, Any]) -> None:
        sidecar = _safe_path(self.path, shard["sidecar"])
        if not sidecar.is_file() or _file_hash(sidecar) != shard["sidecar_sha256"]:
            raise CacheIntegrityError(f"committed sidecar missing or corrupted: {sidecar}")
        data = _load_json(sidecar)
        if (data.get("index") != shard["index"] or data.get("rows") != shard["rows"]
                or len(data.get("entries", [])) != shard["samples"]):
            raise CacheIntegrityError(f"sidecar content disagrees with manifest: {sidecar}")

    def _check_shard(self, shard: Mapping[str, Any]) -> None:
        name = shard["index"]
        for filename, expected in shard["files"].items():
            path = _safe_path(self.path, filename)
            if not path.is_file() or _file_hash(path) != expected:
                raise CacheIntegrityError(f"corrupted shard {name}: {filename}")
            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                if self.layout == "dense" and (arr.shape != (shard["rows"], self.width)
                        or arr.dtype != (np.dtype('uint16') if self.dtype == 'bfloat16' else np.dtype(self.dtype))):
                    raise ValueError("matrix shape/dtype mismatch")
                if self.layout == "csr":
                    if filename.endswith("-indptr.npy") and (arr.shape != (shard["rows"] + 1,) or int(arr[0]) != 0):
                        raise ValueError("CSR row pointer mismatch")
                    if filename.endswith("-indices.npy") and (arr.dtype != np.int32 or (arr.size and (arr.min() < 0 or arr.max() >= self.width))):
                        raise ValueError("invalid CSR feature indices")
                    if filename.endswith("-values.npy") and (arr.dtype != np.float32 or not bool(np.isfinite(arr).all())):
                        raise ValueError("invalid CSR feature values")
            except (OSError, ValueError) as exc:
                raise CacheIntegrityError(f"invalid stored matrix {filename}: {exc}") from exc
        if self.layout == "csr":
            pointers = np.load(_safe_path(self.path, f"shard-{name:05d}-indptr.npy"), mmap_mode="r", allow_pickle=False)
            indices = np.load(_safe_path(self.path, f"shard-{name:05d}-indices.npy"), mmap_mode="r", allow_pickle=False)
            values = np.load(_safe_path(self.path, f"shard-{name:05d}-values.npy"), mmap_mode="r", allow_pickle=False)
            if int(pointers[-1]) != len(indices) or len(indices) != len(values) or bool((np.diff(pointers) < 0).any()):
                raise CacheIntegrityError("invalid CSR offsets or value count")
        self._verified.add(name)

    def _arrays(self, shard: dict[str, Any]) -> Any:
        index = shard["index"]
        if index not in self._mmap:
            if index not in self._verified:
                self._check_shard(shard)
            if self.layout == "dense":
                arrays = np.load(_safe_path(self.path, f"shard-{index:05d}.npy"), mmap_mode="r", allow_pickle=False)
            else:
                arrays = tuple(np.load(_safe_path(self.path, f"shard-{index:05d}-{name}.npy"), mmap_mode="r", allow_pickle=False)
                               for name in ("values", "indices", "indptr"))
            self._mmap[index] = arrays
            # Never retain a mapping for every shard of a large stream.
            while len(self._mmap) > 2:
                self._mmap.popitem(last=False)
        self._mmap.move_to_end(index)
        return self._mmap[index]

    def has(self, sample_id: str) -> bool:
        return sample_id in self._index

    def metadata(self, sample_id: str) -> SampleRecord:
        return SampleRecord.from_dict(self._index[sample_id][1]["record"])

    def get(self, sample_id: str) -> StoredSample:
        shard, entry = self._index[sample_id]
        start, end = int(entry["start"]), int(entry["end"])
        arrays = self._arrays(shard)
        if self.layout == "dense":
            rows = _from_storage(arrays[start:end], self.dtype)
        else:
            values, indices, indptr = arrays
            rows = np.zeros((end - start, self.width), dtype=np.float32)
            for i, r in enumerate(range(start, end)):
                a, b = int(indptr[r]), int(indptr[r+1])
                rows[i, indices[a:b]] = values[a:b]
        return StoredSample(record=SampleRecord.from_dict(entry["record"]),
                            positions=tuple(entry["positions"]), values=rows)

    def iter_samples(self) -> Iterator[StoredSample]:
        for sample_id in self._ordered_ids:
            yield self.get(sample_id)

    def iter_batches(self, *, max_rows: int = 256) -> Iterator[list[StoredSample]]:
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("max_rows must be positive")
        batch: list[StoredSample] = []
        total = 0
        for item in self.iter_samples():
            n = len(item.positions)
            if n > max_rows:
                raise ValueError(f"sample {item.record.sample_id} requires {n} positions > max_rows={max_rows}")
            if batch and total + n > max_rows:
                yield batch
                batch, total = [], 0
            batch.append(item)
            total += n
        if batch:
            yield batch

    @property
    def sample_count(self) -> int:
        return len(self._ordered_ids)

    @property
    def row_count(self) -> int:
        return sum(shard["rows"] for shard in self.manifest["shards"])


class ShardedWriter:
    """Single-writer append-only local store with manifest-atomic shard commits."""

    def __init__(self, path: str | Path, *, identity: dict[str, Any],
                 kind: str, width: int, dtype: str = "float32", layout: str = "dense",
                 shard_target_bytes: int = DEFAULT_SHARD_BYTES, resume: bool = True,
                 repair_corruption: bool = False,
                 max_quantization_error: float | None = None) -> None:
        if kind not in {"activations", "codes"} or layout not in _ALLOWED_LAYOUTS:
            raise ValueError("unknown kind/layout")
        if kind == "activations" and layout != "dense":
            raise ValueError("raw activations must use dense layout")
        if type(width) is not int or width < 1:
            raise ValueError("width must be positive")
        if type(shard_target_bytes) is not int or shard_target_bytes < 128:
            raise ValueError("shard_target_bytes must be >=128")
        if dtype not in {"float32", "float16", "bfloat16"} or (layout == "csr" and dtype != "float32"):
            raise ValueError("CSR codes use float32; dense arrays support float32/float16/bfloat16")
        if max_quantization_error is not None and (not np.isfinite(max_quantization_error) or max_quantization_error < 0):
            raise ValueError("max_quantization_error must be nonnegative finite")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "shards").mkdir(exist_ok=True)
        self._lock_file = (self.path / ".writer.lock").open("a+b")
        try:
            import fcntl
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self._lock_file.close()
            raise RuntimeError("another writer holds this cache directory") from exc
        self._closed = False
        self._pending: list[tuple[SampleRecord, tuple[int, ...], Any]] = []
        self._pending_ids: dict[str, str] = {}
        self._pending_bytes = 0
        self.shard_target_bytes = shard_target_bytes
        self.max_quantization_error = max_quantization_error
        self.max_observed_quantization_error = 0.0
        self._identity = identity
        self._next_index = 0
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            if not resume:
                self.abort()
                raise FileExistsError("cache already exists; open with resume=True")
            existing = _load_json(manifest_path)
            if (existing.get("identity_key") != fingerprint(identity) or existing.get("kind") != kind
                    or existing.get("width") != width or existing.get("dtype") != dtype
                    or existing.get("layout") != layout):
                self.abort()
                raise CacheIntegrityError("cache identity, width, dtype, or layout differs")
            self.manifest = existing
        else:
            if any((self.path / "shards").iterdir()):
                self.abort()
                raise CacheIntegrityError("unclaimed data without a manifest")
            self.manifest = {"schema": SCHEMA, "identity": identity, "identity_key": fingerprint(identity),
                             "kind": kind, "width": width, "dtype": dtype, "layout": layout,
                             "shards": [], "repairs": []}
            _atomic_json(manifest_path, self.manifest)
        if repair_corruption:
            kept = []
            for item in self.manifest["shards"]:
                try:
                    self._check_existing(item)
                    kept.append(item)
                except CacheIntegrityError as exc:
                    self.manifest["repairs"].append({"index": item["index"], "reason": str(exc)})
            if len(kept) != len(self.manifest["shards"]):
                self.manifest["shards"] = kept
                _atomic_json(manifest_path, self.manifest)
        try:
            self._reader = ShardedReader(self.path, verify_shards=True, expected_identity=identity)
        except Exception:
            self.abort()
            raise
        used_indexes = [int(match.group(1)) for f in (self.path / "shards").iterdir()
                        if (match := re.match(r"shard-(\d{5})", f.name))]
        self._next_index = max(used_indexes, default=-1) + 1

    def _check_existing(self, shard: dict[str, Any]) -> None:
        # Verify this one shard with the full consistency checks from reader.
        temp = dict(self.manifest)
        temp["shards"] = [shard]
        # Actual ShardedReader instantiation expects a manifest on disk; validate
        # via the regular reader over the current manifest, but construct helpers.
        checker = object.__new__(ShardedReader)
        checker.path = self.path
        checker.layout, checker.width, checker.dtype = (self.manifest[key] for key in ("layout", "width", "dtype"))
        checker._verified = set()
        checker._check_sidecar(shard)
        checker._check_shard(shard)

    def has(self, record: SampleRecord) -> bool:
        sid = record.sample_id
        if sid in self._pending_ids:
            if self._pending_ids[sid] != record.key:
                raise CacheIntegrityError(f"sample {sid!r} changed input token IDs or split metadata")
            return True
        if self._reader.has(sid):
            if self._reader.metadata(sid).key != record.key:
                raise CacheIntegrityError(f"sample {sid!r} differs from committed record")
            return True
        return False

    def append(self, record: SampleRecord, values: Any) -> bool:
        if self._closed:
            raise RuntimeError("writer already closed")
        if self.has(record):
            return False
        positions = resolve_positions(record)
        matrix, error = _to_storage(values, self.manifest["dtype"],
                                    max_quantization_error=self.max_quantization_error)
        if matrix.shape != (len(positions), self.manifest["width"]):
            raise ValueError(f"activation/code matrix must have shape [{len(positions)}, {self.manifest['width']}]")
        if self.manifest["layout"] == "csr":
            row, col = np.nonzero(matrix)
            data = (np.array(matrix[row, col], dtype=np.float32), np.asarray(col, dtype=np.int32),
                    np.bincount(row, minlength=len(matrix)).astype(np.int64))
            memory = sum(p.nbytes for p in data)
        else:
            data = matrix
            memory = matrix.nbytes
        if memory > self.shard_target_bytes:
            raise ValueError(f"sample {record.sample_id!r} exceeds one shard memory budget; "
                             "reduce token selection or increase shard_target_bytes")
        if self._pending and self._pending_bytes + memory > self.shard_target_bytes:
            self.flush()
        self._pending.append((record, positions, data))
        self._pending_ids[record.sample_id] = record.key
        self._pending_bytes += memory
        self.max_observed_quantization_error = max(self.max_observed_quantization_error, error)
        return True

    def flush(self) -> None:
        if not self._pending:
            return
        index = self._next_index
        self._next_index += 1
        prefix = f"shard-{index:05d}"
        count = sum(len(pos) for _, pos, _ in self._pending)
        entries: list[dict[str, Any]] = []
        cursor = 0
        for record, positions, _ in self._pending:
            n = len(positions)
            entries.append({"record": record.to_dict(), "record_sha256": record.key,
                            "positions": list(positions), "start": cursor, "end": cursor + n})
            cursor += n
        sidecar_path = _safe_path(self.path, f"{prefix}.json")
        _atomic_json(sidecar_path, {"index": index, "rows": count, "entries": entries})
        files: dict[str, str] = {}
        if self.manifest["layout"] == "dense":
            tensor = np.concatenate([matrix for _, _, matrix in self._pending], axis=0)
            p = _safe_path(self.path, f"{prefix}.npy")
            _atomic_npy(p, tensor)
            files[p.name] = _file_hash(p)
        else:
            vals, inds, lengths = [], [], []
            for _, _, (v, i, counts) in self._pending:
                vals.append(v); inds.append(i); lengths.append(counts)
            values = np.concatenate(vals).astype(np.float32, copy=False)
            indices = np.concatenate(inds).astype(np.int32, copy=False)
            indptr = np.empty(count + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(np.concatenate(lengths), out=indptr[1:])
            for label, tensor in (("values", values), ("indices", indices), ("indptr", indptr)):
                p = _safe_path(self.path, f"{prefix}-{label}.npy")
                _atomic_npy(p, tensor)
                files[p.name] = _file_hash(p)
        new = {"index": index, "rows": count, "samples": len(entries),
               "files": files, "sidecar": sidecar_path.name,
               "sidecar_sha256": _file_hash(sidecar_path)}
        self.manifest["shards"].append(new)
        _atomic_json(self.path / "manifest.json", self.manifest)
        # Extend the in-memory index incrementally. Reopening and re-indexing
        # every historical sidecar on each flush would be quadratic in shard count.
        self._reader.manifest = self.manifest
        self._reader._verified.add(index)
        for entry in entries:
            rec = SampleRecord.from_dict(entry["record"])
            self._reader._index[rec.sample_id] = (new, entry)
            self._reader._ordered_ids.append(rec.sample_id)
        self._pending.clear(); self._pending_ids.clear(); self._pending_bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
            # Convenience index is not authoritative. Stream it into a temp
            # file so closing a huge cache never duplicates the full index RAM.
            output = self.path / "samples.jsonl"
            fd, temp = tempfile.mkstemp(prefix=".sample-index-", dir=self.path)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for sid in self._reader._ordered_ids:
                        handle.write(stable_json(self._reader.metadata(sid).to_dict()) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, output)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        finally:
            self.abort()

    def abort(self) -> None:
        if not self._closed:
            self._pending.clear(); self._pending_ids.clear()
            self._lock_file.close()  # OS releases advisory writer lock after crash.
            self._closed = True

    def __enter__(self) -> "ShardedWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    @property
    def committed_samples(self) -> int:
        return self._reader.sample_count


class ActivationWriter(ShardedWriter):
    def __init__(self, cache_root: str | Path, identity: ActivationIdentity, *,
                 width: int, shard_target_bytes: int = DEFAULT_SHARD_BYTES,
                 resume: bool = True, repair_corruption: bool = False,
                 max_quantization_error: float | None = None) -> None:
        self.activation_identity = identity
        super().__init__(Path(cache_root) / identity.key, identity=identity.to_dict(),
                         kind="activations", width=width, dtype=identity.dtype,
                         shard_target_bytes=shard_target_bytes, resume=resume,
                         repair_corruption=repair_corruption,
                         max_quantization_error=max_quantization_error)


class CodeWriter(ShardedWriter):
    def __init__(self, activation_store_path: str | Path, identity: CodeIdentity, *,
                 layout: str = "csr", shard_target_bytes: int = DEFAULT_SHARD_BYTES,
                 resume: bool = True, repair_corruption: bool = False) -> None:
        self.code_identity = identity
        source = ShardedReader(activation_store_path, verify_shards=True)
        if source.kind != "activations" or source.width != identity.d_in:
            raise ValueError("SAE input width does not match activation cache")
        code_id = {"activation_key": source.manifest["identity_key"],
                   "sae": identity.to_dict(), "storage_layout": layout, "storage_dtype": "float32"}
        super().__init__(source.path / "codes" / fingerprint(code_id),
                         identity=code_id, kind="codes", width=identity.d_sae,
                         dtype="float32", layout=layout,
                         shard_target_bytes=shard_target_bytes, resume=resume,
                         repair_corruption=repair_corruption)
