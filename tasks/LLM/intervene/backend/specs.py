"""Versioned, deterministic experiment definitions for M2 interventions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from saebase.api import ActivationSite
from saebase.store import SampleRecord, fingerprint, resolve_positions


@dataclass(frozen=True)
class Condition:
    """A single feature intervention; values are SAE code units, not model units.

    coalition retains the listed feature IDs while zeroing the other selected
    IDs; untouched SAE features and reconstruction residual are preserved.
    reconstruction is the separately identified *full SAE replacement* control.
    """
    name: str
    operation: str
    feature_ids: tuple[int, ...] = ()
    value: float = 0.0
    retained_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("condition name must not be empty")
        allowed = {"identity", "ablate", "scale", "clamp", "add", "coalition", "reconstruction"}
        if self.operation not in allowed:
            raise ValueError(f"unknown intervention operation {self.operation!r}")
        for field_name in ("feature_ids", "retained_ids"):
            ids = getattr(self, field_name)
            if not isinstance(ids, tuple) or len(set(ids)) != len(ids) or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError(f"{field_name} requires distinct, nonnegative integer feature IDs")
        if not isinstance(self.value, (float, int)) or isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("intervention value must be a finite scalar")
        if self.operation in {"identity", "reconstruction"} and (self.feature_ids or self.retained_ids):
            raise ValueError("identity and reconstruction do not select individual features")
        if self.operation not in {"identity", "reconstruction"} and not self.feature_ids:
            raise ValueError("feature intervention needs at least one feature ID")
        if self.operation != "coalition" and self.retained_ids:
            raise ValueError("retained_ids applies only to coalition")
        if self.operation == "coalition" and not set(self.retained_ids).issubset(self.feature_ids):
            raise ValueError("retained_ids must be selected feature IDs")
        if self.operation not in {"clamp", "add", "scale"} and self.value != 0:
            raise ValueError("value applies only to scaling, clamping and additive interventions")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementSample:
    record: SampleRecord
    readout_position: int = -1
    target_token_id: int | None = None
    reference_token_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record, SampleRecord):
            raise TypeError("record must be an M1 SampleRecord")
        if len(resolve_positions(self.record)) != 1:
            raise ValueError("M2 requires exactly one patch position per measurement sample")
        if type(self.readout_position) is not int:
            raise ValueError("readout_position must be an integer")
        mask = self.record.attention_mask or (1,) * len(self.record.token_ids)
        # Current adapters do not accept explicit attention masks; left padding unsupported.
        if 1 in mask and tuple(mask) != (1,) * sum(mask) + (0,) * (len(mask) - sum(mask)):
            raise ValueError("M2 supports unpadded or right-padded inputs only")
        length = sum(mask)
        if not -length <= self.readout_position < length:
            raise ValueError("readout position is outside valid tokens")
        for name in ("target_token_id", "reference_token_id"):
            val = getattr(self, name)
            if val is not None and (type(val) is not int or val < 0):
                raise ValueError(f"{name} must be a nonnegative integer token ID")
        if self.reference_token_id is not None and self.target_token_id is None:
            raise ValueError("target_token_id required when reference_token_id provided")

    @property
    def patch_position(self) -> int:
        return resolve_positions(self.record)[0]

    @property
    def valid_length(self) -> int:
        return sum(self.record.attention_mask or (1,) * len(self.record.token_ids))

    @property
    def readout_index(self) -> int:
        return self.readout_position % self.valid_length

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.record.token_ids[:self.valid_length]

    def to_dict(self) -> dict[str, Any]:
        return {"record": self.record.to_dict(), "readout_position": self.readout_position,
                "target_token_id": self.target_token_id, "reference_token_id": self.reference_token_id}


@dataclass(frozen=True)
class ExperimentIdentity:
    model_state_sha256: str
    sae_state_sha256: str
    site: ActivationSite
    model_key: str = "custom"
    sae_key: str = "custom"
    tokenizer_sha256: str | None = None
    backend: str = "transformer_lens"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_state_sha256", "sae_state_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.tokenizer_sha256 is not None and (len(self.tokenizer_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.tokenizer_sha256)):
            raise ValueError("tokenizer_sha256 must be a SHA-256 digest")
        if not isinstance(self.site, ActivationSite):
            raise TypeError("site must be an ActivationSite")
        fingerprint(self.metadata)

    @property
    def key(self) -> str:
        return fingerprint({"schema": "saebase.m2.experiment.v1", **self.to_dict()})

    def to_dict(self) -> dict[str, Any]:
        return {"model_state_sha256": self.model_state_sha256,
                "sae_state_sha256": self.sae_state_sha256,
                "site": asdict(self.site), "model_key": self.model_key,
                "sae_key": self.sae_key, "tokenizer_sha256": self.tokenizer_sha256,
                "backend": self.backend, "metadata": self.metadata}


def measurement_id(experiment: ExperimentIdentity, sample: MeasurementSample, condition: Condition) -> str:
    return fingerprint({"schema": "saebase.m2.measurement.v1", "experiment": experiment.key,
                        "sample": sample.to_dict(), "condition": condition.to_dict()})
