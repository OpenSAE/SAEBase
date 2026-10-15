"""Immutable grouped evaluation contracts; training seeds are not evaluation seeds."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping
import json
import math

from saebase.store import fingerprint

SCHEMA="saebase.m4.audit.config.v1"

@dataclass(frozen=True)
class AuditConfig:
    evaluation_seeds: tuple[int,...]=(0,1,2,3,4)
    bootstrap_replicates: int=512
    confidence: float=0.95
    sample_sizes: tuple[int,...]=(32,128,512)
    top_k: int=1
    seed_kind: str="evaluation"
    training_seed: int|None=None
    metric_ids: tuple[str,...]=()
    schema: str=SCHEMA

    def __post_init__(self):
        if self.schema!=SCHEMA:
            raise ValueError("incompatible M4 config schema")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds))!=len(self.evaluation_seeds):
            raise ValueError("evaluation seeds must be distinct and nonempty")
        if any(type(s) is not int or s<0 for s in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be nonnegative integers")
        if self.seed_kind!="evaluation":
            raise ValueError("training and evaluation seeds must not be conflated")
        if self.training_seed is not None and (type(self.training_seed) is not int or self.training_seed<0):
            raise ValueError("training_seed must be nonnegative or null")
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates<20:
            raise ValueError("bootstrap_replicates must be at least 20")
        if type(self.confidence) not in (float,int) or not 0<self.confidence<1:
            raise ValueError("confidence must be between 0 and 1")
        if type(self.top_k) is not int or self.top_k<1:
            raise ValueError("top_k must be positive")
        if any(type(x) is not int or x<2 for x in self.sample_sizes):
            raise ValueError("sample_sizes must contain values >= 2")
        if len(set(self.sample_sizes))!=len(self.sample_sizes):
            raise ValueError("duplicate sample_sizes")
        if len(set(self.metric_ids))!=len(self.metric_ids):
            raise ValueError("duplicate metric IDs")
        fingerprint(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str,Any]) -> "AuditConfig":
        allowed=set(cls.__dataclass_fields__)
        if not isinstance(data,dict) or set(data)-allowed:
            raise ValueError(f"unknown audit fields: {sorted(set(data)-allowed) if isinstance(data,dict) else data!r}")
        kw=dict(data)
        for name in ("evaluation_seeds","sample_sizes","metric_ids"):
            if name in kw:
                kw[name]=tuple(kw[name])
        return cls(**kw)

    @classmethod
    def load(cls, path: str|Path) -> "AuditConfig":
        path=Path(path)
        if path.suffix==".json":
            data=json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yml",".yaml"}:
            import yaml
            data=yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("audit config needs JSON or YAML")
        return cls.from_dict(data)

    @property
    def config_id(self)->str:
        return fingerprint(asdict(self))[:24]

    def to_dict(self)->dict[str,Any]:
        return json.loads(json.dumps(asdict(self),allow_nan=False))

@dataclass(frozen=True)
class Observation:
    model_id: str
    sae_id: str
    metric_id: str
    dataset_id: str
    split: str
    prompt_group_id: str
    sample_id: str
    value: float
    condition_id: str|None=None
    measurement_ref: str|None=None
    unit_count: int=1
    seed: int=0

    def __post_init__(self):
        for key in ("model_id","sae_id","metric_id","dataset_id","split","prompt_group_id","sample_id"):
            if not isinstance(getattr(self,key),str) or not getattr(self,key).strip():
                raise ValueError(f"missing {key}; M4 never assumes independent ungrouped prompts")
        if type(self.value) not in (float,int) or not math.isfinite(self.value):
            raise ValueError("observation value must be finite")
        if type(self.unit_count) is not int or self.unit_count<1:
            raise ValueError("unit_count must be positive")
        if type(self.seed) is not int or self.seed<0:
            raise ValueError("evaluation seed must be nonnegative")

    @property
    def series_key(self)->tuple:
        return (self.model_id,self.sae_id,self.metric_id,
                self.dataset_id,self.split,self.condition_id,self.seed)
