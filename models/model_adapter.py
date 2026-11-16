"""Shared construction and lazy metadata for explicitly declared model specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from models import hooked_lm_common as common


@dataclass(frozen=True)
class ModelAdapter:
    """Bind existing model interfaces to a specification and explicit loading choices."""

    spec: common.BaseModelSpec
    backend: str = "load_from_spec"
    options: dict[str, Any] = field(default_factory=dict)
    construct_default_cuda: bool = False
    lazy_backend: str | None = None
    local_alias: tuple[str, str] | None = None

    def get_spec(self) -> common.BaseModelSpec:
        return self.spec

    def load_model(self, device: str | None = None) -> common.HookedLMHandle:
        from models.loading_policy import resolve_device

        device = resolve_device(device)
        options = dict(self.options)
        if self.local_alias:
            local, architecture = self.local_alias
            aliases = dict(options.get("architecture_aliases", {}))
            if Path(local).joinpath("config.json").is_file():
                aliases[local] = architecture
            options["architecture_aliases"] = aliases
        if self.backend == "load_from_spec":
            return common.load_from_spec(self.spec, device=device, **options)
        if self.backend == "load_hf_hooks_model":
            from models.hf_hooks_backend import load_hf_hooks_model

            return load_hf_hooks_model(
                self.spec.model_name,
                key=self.spec.key,
                family=self.spec.family,
                device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
                **options,
            )
        if self.backend == "load_hf_logits_model":
            from models.hf_logits_backend import load_hf_logits_model

            return load_hf_logits_model(
                self.spec.model_name,
                key=self.spec.key,
                family=self.spec.family,
                device=device or "cuda",
                **options,
            )
        raise ValueError("Unknown model backend: " + self.backend)

    def construct(self, cfg: DictConfig) -> common.HookedLMHandle:
        device = cfg.model.get("device")
        return self.load_model(device=device)

    def construct_lazy(self, cfg: DictConfig) -> dict[str, Any]:
        from models.loading_policy import resolve_device

        result = {
            "key": self.spec.key,
            "model_name": self.spec.model_name,
            "family": self.spec.family,
            "lazy": True,
            "device": resolve_device(cfg.model.get("device")),
        }
        if self.lazy_backend:
            result["backend"] = self.lazy_backend
        return result
