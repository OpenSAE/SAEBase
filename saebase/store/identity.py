"""Versioned identities and exact token-site selection for activation caches.

Model weights/tokenizer/preprocessing are part of activation identity. SAE
checkpoint/inference metadata is *only* part of the separately keyed code cache.
The SHA-256 hashes are deterministic identifiers, not an authenticity guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import weakref
from typing import Any, Mapping

import torch

_SHA = re.compile(r"^[0-9a-f]{64}$")
# Repeated fingerprints of one loaded module skip the full byte hash when the
# storage address, layout, and in-place version of every tensor are unchanged.
_STATE_SHA_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _state_signature(items: list[tuple[str, torch.Tensor]]) -> tuple:
    return tuple(
        (
            name,
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            int(value.data_ptr()),
            int(value.storage_offset()),
            int(getattr(value, "_version", 0)),
        )
        for name, value in items
    )


def sha256_state_dict(module: Any) -> str:
    """Hash the complete sorted loaded state, identical to M0's reference format.

    The digest is cached on the module object. A later call returns that digest
    only when every tensor still has the same storage, layout, and in-place
    version, so an in-place weight edit is hashed again.
    """
    if not callable(getattr(module, "state_dict", None)):
        raise TypeError("module must expose state_dict() for checkpoint verification")
    state = module.state_dict()
    if not state:
        raise ValueError("cannot fingerprint an empty state_dict")
    items: list[tuple[str, torch.Tensor]] = []
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor state_dict value: {name!r}")
        items.append((name, value))
    signature = _state_signature(items)
    try:
        cached = _STATE_SHA_CACHE.get(module)
    except TypeError:
        cached = None
    if isinstance(cached, tuple) and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    for name, value in items:
        tensor = value.detach().contiguous().cpu()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                 separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    result = digest.hexdigest()
    try:
        _STATE_SHA_CACHE[module] = (signature, result)
    except TypeError:
        pass
    return result


def _sha(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 digest")


def _json_mapping(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise TypeError(f"{label} must be a dictionary with string keys")
    stable_json(value)


@dataclass(frozen=True)
class ActivationIdentity:
    """Immutable model/site/precision identity; deliberately SAE-independent.

    model_state_sha256 attests the *full loaded weight state*. tokenizer_sha256
    attests the tokenizer files/configuration chosen by the caller. A mutable
    Hugging Face branch name is insufficient for a reusable production cache.
    """

    model_key: str
    model_state_sha256: str
    tokenizer_sha256: str
    hook_name: str
    layer: int
    dtype: str = "float32"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    chat_template_sha256: str | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_key", "hook_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if self.hook_name.startswith("blocks."):
            parts = self.hook_name.split(".")
            if len(parts) > 2 and parts[1].isdigit() and int(parts[1]) != self.layer:
                raise ValueError("hook_name and layer disagree")
        _sha(self.model_state_sha256, "model_state_sha256")
        _sha(self.tokenizer_sha256, "tokenizer_sha256")
        if self.chat_template_sha256 is not None:
            _sha(self.chat_template_sha256, "chat_template_sha256")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        _json_mapping(self.preprocessing, "preprocessing")
        _json_mapping(self.execution, "execution")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return fingerprint({"schema": "saebase.activation_identity.v1", **self.to_dict()})


@dataclass(frozen=True)
class CodeIdentity:
    """SAE-specific code key, with explicit inference and normalization details."""

    sae_key: str
    sae_state_sha256: str
    d_in: int
    d_sae: int
    inference_mode: str = "per_token"
    normalization: dict[str, Any] = field(default_factory=dict)
    checkpoint_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sae_key, str) or not self.sae_key.strip():
            raise ValueError("sae_key must not be empty")
        _sha(self.sae_state_sha256, "sae_state_sha256")
        if type(self.d_in) is not int or self.d_in < 1 or type(self.d_sae) is not int or self.d_sae < 1:
            raise ValueError("SAE dimensions must be positive integers")
        if self.inference_mode not in {"per_token", "fixed_threshold"}:
            raise ValueError("code caching needs deterministic per-token or fixed-threshold inference")
        _json_mapping(self.normalization, "normalization")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return fingerprint({"schema": "saebase.code_identity.v1", **self.to_dict()})


@dataclass(frozen=True)
class TokenSelection:
    """Meaningful activation site relative to the *stored input token IDs*."""

    mode: str = "last_valid"
    indices: tuple[int, ...] = ()
    char_span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"last_valid", "indices", "all_valid", "span"}:
            raise ValueError("unknown token selection mode")
        if self.mode == "indices":
            if not self.indices or any(type(i) is not int or i < 0 for i in self.indices):
                raise ValueError("indices must be nonempty nonnegative token indices")
            if len(set(self.indices)) != len(self.indices):
                raise ValueError("token indices must be unique")
        elif self.indices:
            raise ValueError("indices are supported only in indices mode")
        if self.mode == "span":
            if (not isinstance(self.char_span, tuple) or len(self.char_span) != 2
                    or any(type(x) is not int for x in self.char_span)
                    or not 0 <= self.char_span[0] < self.char_span[1]):
                raise ValueError("char_span must contain [start, end) offsets")
        elif self.char_span is not None:
            raise ValueError("char_span is supported only in span mode")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "indices": list(self.indices),
                "char_span": list(self.char_span) if self.char_span else None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenSelection":
        return cls(mode=data["mode"], indices=tuple(data.get("indices") or ()),
                   char_span=tuple(data["char_span"]) if data.get("char_span") else None)


@dataclass(frozen=True)
class SampleRecord:
    """A sample's exact original IDs and split/group lineage; no text required."""

    sample_id: str
    token_ids: tuple[int, ...]
    selection: TokenSelection = field(default_factory=TokenSelection)
    attention_mask: tuple[int, ...] | None = None
    prompt_group_id: str | None = None
    split: str = "unspecified"
    variant_role: str = "clean"
    token_offsets: tuple[tuple[int, int], ...] | None = None
    source_activation_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not isinstance(self.selection, TokenSelection):
            raise TypeError("selection must be TokenSelection")
        if not self.token_ids or any(type(v) is not int or v < 0 for v in self.token_ids):
            raise ValueError("token_ids must be nonempty nonnegative integers")
        if self.attention_mask is not None and (len(self.attention_mask) != len(self.token_ids)
                or any(type(v) is not int or v not in (0, 1) for v in self.attention_mask)):
            raise ValueError("attention_mask must contain 0/1 for each token")
        if self.token_offsets is not None and (len(self.token_offsets) != len(self.token_ids)
                or any(len(p) != 2 or any(type(x) is not int for x in p)
                       or p[0] < 0 or p[1] < p[0] for p in self.token_offsets)):
            raise ValueError("token_offsets must contain nonnegative [start, end) pairs per token")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must not be empty")
        if not isinstance(self.variant_role, str) or not self.variant_role.strip():
            raise ValueError("variant_role must not be empty")
        if self.prompt_group_id is not None and not isinstance(self.prompt_group_id, str):
            raise TypeError("prompt_group_id must be a string or None")
        if self.source_activation_sha256 is not None:
            _sha(self.source_activation_sha256, "source_activation_sha256")
        resolve_positions(self)

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "token_ids": list(self.token_ids),
                "selection": self.selection.to_dict(),
                "attention_mask": list(self.attention_mask) if self.attention_mask is not None else None,
                "prompt_group_id": self.prompt_group_id, "split": self.split,
                "variant_role": self.variant_role,
                "token_offsets": [list(p) for p in self.token_offsets] if self.token_offsets is not None else None,
                "source_activation_sha256": self.source_activation_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SampleRecord":
        return cls(sample_id=data["sample_id"], token_ids=tuple(data["token_ids"]),
                   selection=TokenSelection.from_dict(data["selection"]),
                   attention_mask=tuple(data["attention_mask"]) if data.get("attention_mask") is not None else None,
                   prompt_group_id=data.get("prompt_group_id"), split=data.get("split", "unspecified"),
                   variant_role=data.get("variant_role", "clean"),
                   token_offsets=tuple(map(tuple, data["token_offsets"])) if data.get("token_offsets") is not None else None,
                   source_activation_sha256=data.get("source_activation_sha256"))

    @property
    def key(self) -> str:
        return fingerprint({"schema": "saebase.sample.v1", **self.to_dict()})


def resolve_positions(record: SampleRecord) -> tuple[int, ...]:
    """Convert a token policy to unambiguous positions (special tokens excluded in spans)."""
    mask = record.attention_mask or (1,) * len(record.token_ids)
    valid = tuple(i for i, v in enumerate(mask) if v)
    if not valid:
        raise ValueError("sample has no valid tokens")
    policy = record.selection
    if policy.mode == "last_valid":
        return (valid[-1],)
    if policy.mode == "all_valid":
        return valid
    if policy.mode == "indices":
        if any(i >= len(mask) or not mask[i] for i in policy.indices):
            raise ValueError("selected token index is outside valid attention positions")
        return policy.indices
    if record.token_offsets is None:
        raise ValueError("span token alignment requires token_offsets or explicit indices")
    start, end = policy.char_span  # type: ignore[misc]
    chosen = tuple(i for i, (a, b) in enumerate(record.token_offsets)
                   if mask[i] and a < end and b > start and a < b)
    if not chosen:
        raise ValueError("requested char span has no matching valid tokens")
    return chosen
