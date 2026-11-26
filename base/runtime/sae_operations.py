"""Seven real SAEBase Task entrypoints using configured Model/SAE resources.

No model checkpoints are downloaded until task.execute=true. Every returned
measurement originates in upstream SAEBase code; unavailable resources raise.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value

from contextlib import contextmanager
from contextvars import ContextVar

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass

from saebase.registry import TaskModule
from saebase.results import EvaluationResult


def input_path(value: Any) -> Path:
    """Resolve input paths against the original launch directory under Hydra."""
    from hydra.utils import to_absolute_path

    return Path(to_absolute_path(str(value))).expanduser().resolve()


def load_requested_extensions(cfg: Any) -> None:
    from saebase.extensions.registry import ensure_extensions

    names = cfg.task.get("extensions", [])
    if isinstance(names, str):
        raise ValueError("task.extensions must be a list of explicit entry-point names")
    ensure_extensions(names)


def validate_fields(cfg: Any, *, integers=(), numbers=(), booleans=()) -> None:
    """Small input-boundary checks; each Task declares the fields it owns."""
    task = cfg.task
    owner = task.get("name")
    own_fields = task.get(owner, {}) if isinstance(owner, str) else {}

    def label(field):
        return f"task.{owner}.{field}" if field in own_fields else f"task.{field}"

    for field, lower in integers:
        value = task_value(task, field)
        if value is not None and (type(value) is not int or value < lower):
            raise ValueError(f"{label(field)} must be an integer >= {lower}")
    for field, lower in numbers:
        value = task_value(task, field)
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (lower is not None and value < lower)
        ):
            raise ValueError(f"{label(field)} must be a finite valid number")
    for field in booleans:
        if type(task_value(task, field, False)) is not bool:
            raise ValueError(f"{label(field)} must be a boolean")


def validate_sae_selection(cfg: Any) -> None:
    task = cfg.task
    validate_fields(cfg, integers=(("layer", 0),))
    if task.get("sae_key") or cfg.model.get("default_saes") or task.get("execute", False):
        resolve_sae(cfg.model, task.get("sae_key"), task.get("layer"), operation=str(task.name))


def validate_native_selection(cfg: Any) -> None:
    if cfg.task.get("sae_key") is not None or cfg.task.get("layer") is not None:
        raise ValueError(
            f"{cfg.task.name} selects resources in its native protocol, not task.sae_key/layer"
        )


def validate_task_cfg(cfg: Any) -> None:
    """Delegate method rules to their owner; this module only shares resource operations."""
    from importlib import import_module

    from saebase.registry import get_task_plugin

    load_requested_extensions(cfg)
    get_task_plugin(str(cfg.task.name))
    validate_fields(cfg, booleans=("execute",))
    import_module(f"tasks.{cfg.task.name}.operation").validate_cfg(cfg)


def _task(cfg: Any, key: str, default: Any = None) -> Any:
    return task_value(cfg.task, key, default)


@dataclass(frozen=True)
class SelectedSAE:
    """Resource selection, not proof that checkpoint contents have been verified."""

    model_key: str
    sae_key: str
    layer: int
    hook: str
    provider: str
    sae_source: str
    layer_source: str
    required_capabilities: tuple[str, ...]


def resolve_sae(model_cfg: Any, key=None, layer=None, *, operation="capture") -> SelectedSAE:
    """Resolve exactly the same resource for task execution and precheck without weights."""
    from saebase.api import UnsupportedCapability
    from saebase.extensions.registry import external_model, external_sae

    model = str(model_cfg.name)
    key_source = "explicit"
    default_entry = None
    if not key:
        defaults = list(model_cfg.get("default_saes", []))
        if not defaults:
            raise ValueError(f"Model {model!r} has no default SAE; provide task.sae_key")
        default_entry = defaults[0]
        key = (
            default_entry.get("sae", default_entry.get("key"))
            if hasattr(default_entry, "get")
            else default_entry
        )
        key_source = "model.default_saes"
    key = str(key)
    required = (
        ("capture", "patch", "forward") if operation in {"intervene", "generate"} else ("capture",)
    )
    external = external_model(model)
    resource = external_sae(key)
    if external is not None:
        if operation not in {"capture", "intervene", "precheck"}:
            raise UnsupportedCapability(
                f"External provider needs a task-specific {operation} backend"
            )
        if resource is None or resource.model_key != model:
            raise ValueError("External SAE and model must be explicitly registered together")
        layers, hook = (resource.site.layer,), resource.site.hook_name
        if external.capabilities is not None:
            for capability in required:
                if capability not in external.capabilities:
                    raise UnsupportedCapability(
                        f"External model lacks {capability} for {operation}"
                    )
        provider = "external"
    else:
        from saebase.api.sae_profiles import get_profile

        profile = get_profile(key)
        available = model_cfg.get("sae_profiles")
        if available is not None and key not in available:
            raise ValueError(f"SAE {key!r} is not in model {model!r} sae_profiles")
        if model not in profile.allowed_model_keys():
            raise ValueError(f"SAE {key!r} is incompatible with model {model!r}")
        layers, hook, provider = profile.layers, None, "builtin"
    layer_source = "explicit"
    if layer is None and hasattr(default_entry, "get"):
        layer = default_entry.get("layer")
        layer_source = "model.default_saes"
    if layer is None:
        layer = model_cfg.get("default_layers", {}).get(key)
        layer_source = "model.default_layers"
    if layer is None:
        if len(layers) != 1:
            raise ValueError(
                f"SAE {key!r} has multiple layers; provide an explicit layer or model.default_layers"
            )
        layer, layer_source = layers[0], "single_available_layer"
    if type(layer) is not int or layer not in layers:
        raise ValueError(f"Invalid layer {layer!r} for SAE {key!r}; available: {list(layers)}")
    if provider == "builtin":
        hook = profile.hook_name(layer)
    return SelectedSAE(model, key, layer, hook, provider, key_source, layer_source, required)


# Explicit, scoped reuse for read-only observation batches. No implicit global model cache.

_ACTIVE_PAIR = ContextVar("saebase_active_research_pair", default=None)
_PAIR_FINGERPRINTS = ContextVar("saebase_pair_fingerprints", default=None)


@contextmanager
def loaded_pair_session(pair):
    """Reuse a real pair and the Store's verified lease, with a final byte recheck."""
    if _ACTIVE_PAIR.get() is pair:
        # Nested task/dataset scopes reuse the outer resource identity and export.
        yield
        return
    token = _ACTIVE_PAIR.set(pair)
    fingerprints = _PAIR_FINGERPRINTS.set({})
    try:
        from saebase.api.neuronpedia import annotation_session
        from saebase.runtime.performance import policy_scope

        with annotation_session(), policy_scope():
            yield
        lease = _PAIR_FINGERPRINTS.get().get("lease")
        if lease is not None:
            lease.strict_recheck()
    finally:
        _ACTIVE_PAIR.reset(token)
        _PAIR_FINGERPRINTS.reset(fingerprints)


def pair_state_fingerprints(pair):
    """Reuse the existing checkpoint lease only inside an explicit read-only batch."""
    from saebase.store import sha256_state_dict, VerifiedCheckpointSession

    model = getattr(pair.model, "model", pair.model)
    cache = _PAIR_FINGERPRINTS.get() if _ACTIVE_PAIR.get() is pair else None
    if cache is None:
        return sha256_state_dict(model), sha256_state_dict(pair.sae)
    if "lease" not in cache:
        cache["lease"] = VerifiedCheckpointSession.mint(model, pair.sae)
    lease = cache["lease"]
    return lease.leased_sha256(model), lease.leased_sha256(pair.sae)


def load_selection(selection: SelectedSAE, device: str):
    """Load the selected provider and verify its reported identity before task work."""
    from saebase.api import load_registered
    from saebase.extensions import load_external_pair
    from models.loading_policy import resolve_device

    device = resolve_device(device)

    loader = load_external_pair if selection.provider == "external" else load_registered
    pair = _ACTIVE_PAIR.get()
    if pair is None:
        pair = loader(
            model_key=selection.model_key,
            sae_key=selection.sae_key,
            layer=selection.layer,
            device=device,
            require_patch="patch" in selection.required_capabilities,
        )
    if (
        pair.identity.model_key,
        pair.identity.sae_key,
        pair.identity.layer,
        pair.identity.hook_name,
    ) != (selection.model_key, selection.sae_key, selection.layer, selection.hook):
        raise ValueError("Loaded model/SAE/site identity differs from resolved selection")
    for capability in selection.required_capabilities:
        pair.model.capabilities.require(capability)
    return pair


def _require_context(cfg: Any, operation: str) -> None:
    from saebase.registry import get_dataset_plugin, get_model_plugin, get_task_plugin

    if operation == "precheck" or str(cfg.task.name) != operation:
        raise ValueError("Task name must match its registered operation")
    get_task_plugin(operation)
    if operation == "benchmark" and (
        _task(cfg, "benchmark_config") or _task(cfg, "protocol") is not None
    ):
        return
    from saebase.extensions.registry import external_model

    if external_model(str(cfg.model.name)) is not None:
        resolve_sae(cfg.model, _task(cfg, "sae_key"), _task(cfg, "layer"), operation=operation)
        return
    if cfg.dataset.get("samples") is not None:
        # Explicit token samples are already parsed by their Task protocol.
        return
    data = get_dataset_plugin(str(cfg.dataset.name))
    if (
        operation in {"capture", "intervene", "generate", "circuits"}
        and not str(_task(cfg, "prompt", "") or "").strip()
    ):
        data.assert_task_supported(cfg)
    get_model_plugin(str(cfg.model.name)).assert_compatible(cfg)


def _pair(cfg: Any):
    selection = resolve_sae(
        cfg.model, _task(cfg, "sae_key"), _task(cfg, "layer"), operation=str(cfg.task.name)
    )
    return load_selection(selection, str(_task(cfg, "device", "cpu")))


def _prompt(cfg: Any) -> str:
    value = str(_task(cfg, "prompt", "") or "").strip()
    if not value:
        from saebase.datasets.prompts import read_first_prompt

        value = read_first_prompt(cfg.dataset, model_key=str(cfg.model.name))
    if not value:
        raise ValueError("Use a nonempty prompt")
    return value


def _fingerprint_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _maybe_save(operation: str, payload: dict[str, Any], cfg=None) -> dict[str, Any]:
    if payload.get("status") == "configured_only":
        return payload
    outdir = Path.cwd() / "metadata"
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{operation}_result.json"
    if dest.exists():
        from saebase.runtime.experiment_paths import run_mode

        recovering = cfg is not None and run_mode(cfg) == "resume"
        same = json.loads(dest.read_text()) == json.loads(json.dumps(payload, allow_nan=False))
        if recovering and not same and task_value(cfg.task, "feature_selection") == "capture_top":
            import time

            history = outdir / "resume_results"
            history.mkdir(exist_ok=True)
            (history / f"{operation}_{time.time_ns()}.json").write_bytes(dest.read_bytes())
            from saebase.runtime.experiment_results import write_json

            write_json(dest, payload)
        elif not recovering or not same:
            raise FileExistsError("A prior task result already exists or differs: " + str(dest))
        return {**payload, "result_path": f"metadata/{operation}_result.json"}
    from saebase.runtime.experiment_results import write_json

    write_json(dest, payload)
    return {**payload, "result_path": f"metadata/{operation}_result.json"}


def execute_task(
    operation: str,
    cfg: Any,
    *,
    generator: Callable[[list[str]], str] | None = None,
    predictor: Callable[[str, str], bool] | None = None,
    validated: bool = False,
) -> dict[str, Any]:
    """Dispatch to the task-owned execution module after selection checks."""
    from importlib import import_module

    if not validated:
        validate_task_cfg(cfg)
    _require_context(cfg, operation)
    if not bool(_task(cfg, "execute", False)):
        return {
            "status": "configured_only",
            "operation": operation,
            "model_key": str(cfg.model.name),
            "dataset_key": str(cfg.dataset.name),
            "description": "Set task.execute=true and supply required runtime inputs to execute.",
        }
    module = import_module(f"tasks.{operation}.operation")
    if operation == "explain":
        result = module.execute(cfg, generator=generator, predictor=predictor)
    else:
        result = module.execute(cfg)
    if operation in {"capture", "intervene", "generate"}:
        from dataclasses import asdict

        result["selection"] = asdict(
            resolve_sae(cfg.model, _task(cfg, "sae_key"), _task(cfg, "layer"), operation=operation)
        )
    return _maybe_save(operation, result, cfg)


class SAEOperationTask(TaskModule):
    """Runtime-neutral task adapter; all algorithms live in their task package."""

    OPERATION = ""

    def validate_cfg(self, cfg: Any) -> None:
        validate_task_cfg(cfg)

    @property
    def name(self) -> str:
        return self.OPERATION

    def run(self, cfg: Any) -> EvaluationResult:
        self.validate_cfg(cfg)
        return self.run_validated(cfg)

    def run_validated(self, cfg: Any) -> EvaluationResult:
        return EvaluationResult.from_payload(
            self.OPERATION, cfg, execute_task(self.OPERATION, cfg, validated=True)
        )

    def prepare_execution(self, cfg: Any) -> dict:
        """Prepare native protocol inputs without executing the research task."""
        from importlib import import_module

        module = import_module(f"tasks.{self.OPERATION}.operation")
        return getattr(module, "prepare_execution", super().prepare_execution)(cfg)
