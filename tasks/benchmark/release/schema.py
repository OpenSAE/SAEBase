"""Strict, immutable and reproducible M5 release matrix schema."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from saebase.api import ActivationSite
from saebase.sae.registry import get_profile
from saebase.store import fingerprint
from tasks.LLM.benchmark.backend.registry import get_metric
import json

MATRIX_SCHEMA = "saebase.m5.matrix.v1"
STATUS_SCHEMA = "saebase.m5.results.v1"


@dataclass(frozen=True)
class MatrixTarget:
    model_key: str
    family: str
    site: ActivationSite
    sae_keys: tuple[str, ...]
    model_card: str
    sae_cards: dict[str, str]
    usage_note: str = "Downloaded on demand; checkpoint weights are not redistributed."

    def __post_init__(self) -> None:
        if not self.model_key or not self.family or len(self.sae_keys) < 2:
            raise ValueError("each model family needs at least two SAE checkpoints")
        if len(set(self.sae_keys)) != len(self.sae_keys):
            raise ValueError("SAE checkpoint keys must be distinct")
        if not self.model_card.startswith("https://") or not self.usage_note:
            raise ValueError("source model card and usage note required")
        for sae_key in self.sae_keys:
            prof = get_profile(sae_key)
            if self.model_key not in prof.allowed_model_keys():
                raise ValueError(f"SAE {sae_key} cannot attach to model {self.model_key}")
            if prof.hook_name(self.site.layer) != self.site.hook_name:
                raise ValueError(f"SAE {sae_key} does not share chosen hook {self.site.hook_name}")
            if not self.sae_cards.get(sae_key, "").startswith("https://"):
                raise ValueError(f"SAE {sae_key} needs a public source/model card")

    def to_dict(self):
        return {
            "model_key": self.model_key,
            "family": self.family,
            "site": {
                "layer": self.site.layer,
                "hook_name": self.site.hook_name,
                "position": self.site.position,
            },
            "sae_keys": list(self.sae_keys),
            "model_card": self.model_card,
            "sae_cards": self.sae_cards,
            "usage_note": self.usage_note,
        }


@dataclass(frozen=True)
class MatrixConfig:
    name: str
    targets: tuple[MatrixTarget, ...]
    prompts: tuple[str, ...]
    metrics: tuple[str, ...]
    conditions: tuple[dict[str, Any], ...]
    seed: int = 0
    max_measurements: int = 256
    max_runtime_seconds: int = 2400
    max_token_count: int = 80
    bootstrap_replicates: int = 64
    version: str = "0.2.0"
    status_schema: str = STATUS_SCHEMA
    schema: str = MATRIX_SCHEMA

    def __post_init__(self):
        if self.schema != MATRIX_SCHEMA or self.status_schema != STATUS_SCHEMA:
            raise ValueError("invalid M5 matrix or output schema")
        if self.version != "0.2.0":
            raise ValueError("M5 release version must be 0.2.0")
        if not self.name or len(self.targets) < 2 or len(self.prompts) < 3:
            raise ValueError("M5 requires two models and at least three independent prompt groups")
        if len({t.model_key for t in self.targets}) != len(self.targets):
            raise ValueError("duplicate model keys")
        if len({t.family for t in self.targets}) < 2:
            raise ValueError("M5 requires at least two independent model families")
        if any(not p.strip() for p in self.prompts) or len(set(self.prompts)) != len(self.prompts):
            raise ValueError("prompts must be nonempty distinct texts")
        if len(set(self.metrics)) != len(self.metrics) or not self.metrics:
            raise ValueError("duplicate or empty metrics")
        for key in self.metrics:
            get_metric(key)
        if not self.conditions or len({c.get("name") for c in self.conditions}) != len(
            self.conditions
        ):
            raise ValueError("unique intervention conditions required")
        from tasks.LLM.intervene.backend import Condition

        for cond in self.conditions:
            Condition(
                name=cond["name"],
                operation=cond["operation"],
                feature_ids=tuple(cond.get("feature_ids", [])),
                retained_ids=tuple(cond.get("retained_ids", [])),
                value=cond.get("value", 0.0),
            )
        for k in (
            "max_measurements",
            "max_runtime_seconds",
            "max_token_count",
            "bootstrap_replicates",
        ):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(f"{k} must be a positive integer")
        if self.bootstrap_replicates < 20:
            raise ValueError("bootstrap_replicates must be >=20")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            len(self.targets)
            * len(self.prompts)
            * len(self.conditions)
            * max(len(t.sae_keys) for t in self.targets)
            > 100000
        ):
            raise ValueError("matrix dimensions exceed safety limit")
        fingerprint(self.to_dict())

    def to_dict(self):
        return {
            "schema": self.schema,
            "status_schema": self.status_schema,
            "name": self.name,
            "targets": [t.to_dict() for t in self.targets],
            "prompts": list(self.prompts),
            "metrics": list(self.metrics),
            "conditions": list(self.conditions),
            "seed": self.seed,
            "max_measurements": self.max_measurements,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_token_count": self.max_token_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "version": self.version,
        }

    @property
    def matrix_id(self):
        return fingerprint(self.to_dict())[:24]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatrixConfig":
        allowed = {
            "schema",
            "status_schema",
            "name",
            "targets",
            "prompts",
            "metrics",
            "conditions",
            "seed",
            "max_measurements",
            "max_runtime_seconds",
            "max_token_count",
            "bootstrap_replicates",
            "version",
        }
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("unknown M5 matrix fields")
        groups = []
        for target in data["targets"]:
            keys = {
                "model_key",
                "family",
                "site",
                "sae_keys",
                "model_card",
                "sae_cards",
                "usage_note",
            }
            if set(target) - keys:
                raise ValueError("unknown M5 target fields")
            groups.append(
                MatrixTarget(
                    target["model_key"],
                    target["family"],
                    ActivationSite(**target["site"]),
                    tuple(target["sae_keys"]),
                    target["model_card"],
                    target["sae_cards"],
                    target.get(
                        "usage_note",
                        "Downloaded on demand; checkpoint weights are not redistributed.",
                    ),
                )
            )
        return cls(
            name=data["name"],
            targets=tuple(groups),
            prompts=tuple(data["prompts"]),
            metrics=tuple(data["metrics"]),
            conditions=tuple(data["conditions"]),
            seed=data.get("seed", 0),
            max_measurements=data.get("max_measurements", 256),
            max_runtime_seconds=data.get("max_runtime_seconds", 2400),
            max_token_count=data.get("max_token_count", 80),
            bootstrap_replicates=data.get("bootstrap_replicates", 64),
            version=data.get("version", "0.2.0"),
            schema=data.get("schema"),
            status_schema=data.get("status_schema", STATUS_SCHEMA),
        )


def load_matrix(path: str | Path) -> MatrixConfig:
    file = Path(path)
    if file.suffix == ".json":
        obj = json.loads(file.read_text(encoding="utf-8"))
    elif file.suffix in {".yml", ".yaml"}:
        import yaml

        obj = yaml.safe_load(file.read_text(encoding="utf-8"))
    else:
        raise ValueError("M5 matrix requires JSON/YAML")
    return MatrixConfig.from_dict(obj)
