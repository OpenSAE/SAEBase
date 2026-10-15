"""Bridge SAEBase capture/code caches to the original SAE Input Localization estimators.

The mathematical estimators live in the independent sae_localize package.
This module prepares complete Boolean input grids and preserves their lineage.
It does not turn SAE feature ablation into an input-factor intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from saebase.api import ActivationSite, ModelAdapter, SAEIdentity, capture_features
from saebase.data.controlled_inputs import (
    ControlledInputDesign, ControlledInputManifest, load_controlled_input_manifest,
)


def _ordered_states(values: Mapping[int, str], n_factors: int, *, label: str) -> tuple[str, ...]:
    """Require one explicitly indexed item for every Boolean input state."""
    if type(n_factors) is not int or n_factors < 1:
        raise ValueError("n_factors must be a positive integer")
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must map integer Boolean state masks to strings")
    count = 1 << n_factors
    if len(values) != count or any(type(mask) is not int for mask in values) or set(values) != set(range(count)):
        raise ValueError(f"{label} must have exactly the state keys 0..{count - 1}")
    ordered = tuple(values[i] for i in range(count))
    if any(not isinstance(v, str) or not v.strip() for v in ordered):
        raise ValueError(f"{label} must contain nonempty strings")
    return ordered


@dataclass(frozen=True)
class LocalizationGrid:
    """Frozen SAE codes in full, uniform Boolean bitmask order.

    State i has factor k set iff bit k of i is one. Declaring the keys does
    not prove that supplied prompts implement the intended factor changes.
    """

    codes: np.ndarray
    n_factors: int
    state_sample_ids: tuple[str, ...]
    source: str
    site: ActivationSite | None = None
    identity: SAEIdentity | None = None
    activations: np.ndarray | None = None
    design_key: str | None = None

    def __post_init__(self) -> None:
        if type(self.n_factors) is not int or self.n_factors < 1:
            raise ValueError("n_factors must be a positive integer")
        z = np.asarray(self.codes)
        if (z.ndim != 2 or z.shape[0] != (1 << self.n_factors)
                or z.shape[1] < 1 or not np.issubdtype(z.dtype, np.number)
                or not np.isfinite(z).all()):
            raise ValueError("codes must be finite [2**n_factors, n_features] data")
        if (len(self.state_sample_ids) != len(z)
                or len(set(self.state_sample_ids)) != len(z)
                or any(not isinstance(s, str) or not s.strip() for s in self.state_sample_ids)):
            raise ValueError("state_sample_ids must identify every input state uniquely")
        if self.activations is not None:
            h = np.asarray(self.activations)
            if h.ndim != 2 or len(h) != len(z) or not np.isfinite(h).all():
                raise ValueError("activations must have finite matching grid rows")
        if self.identity is not None:
            if z.shape[1] != self.identity.d_sae:
                raise ValueError("SAE identity/code dimension mismatch")
            if self.site is not None and (self.site.hook_name != self.identity.hook_name
                                           or self.site.layer != self.identity.layer):
                raise ValueError("SAE identity site differs from requested site")

    @property
    def n_features(self) -> int:
        return int(self.codes.shape[1])


@dataclass(frozen=True)
class LocalizationResult:
    """Original Localize scope curve with explicit protocol and provenance."""

    curve: np.ndarray
    assignment: np.ndarray
    budget: int
    total_variance: float
    relative_omission: float
    grid: LocalizationGrid
    estimator_version: str | None = None
    exact_source_sha256: str | None = None
    estimated_curve: np.ndarray | None = None
    pairs_per_scope: int | None = None
    seed: int | None = None

    @property
    def normalized_curve(self) -> np.ndarray | None:
        """Undefined for an activation grid with zero total variance."""
        if self.total_variance <= 0:
            return None
        return self.curve / self.total_variance

    def summary(self) -> dict[str, Any]:
        """JSON-safe metadata; full curves and scope masks remain in the arrays."""
        site = self.grid.site
        return {
            "n_factors": self.grid.n_factors,
            "n_states": int(self.grid.codes.shape[0]),
            "n_features": self.grid.n_features,
            "relative_omission": self.relative_omission,
            "total_variance": self.total_variance,
            "scope_budget": self.budget,
            "assignment_cost": sum(int(mask).bit_count() for mask in self.assignment),
            "source": self.grid.source,
            "design_key": self.grid.design_key,
            "site": ({"layer": site.layer, "hook_name": site.hook_name,
                       "position": site.position} if site is not None else None),
            "identity": self.grid.identity.to_dict() if self.grid.identity else None,
            "estimator_version": self.estimator_version,
            "exact_source_sha256": self.exact_source_sha256,
            "paired_estimate": self.estimated_curve is not None,
            "pairs_per_scope": self.pairs_per_scope,
            "seed": self.seed,
        }


def collect_localization_grid(
    model: ModelAdapter,
    sae: Any,
    *,
    site: ActivationSite,
    prompts_by_state: Mapping[int, str] | None = None,
    n_factors: int | None = None,
    identity: SAEIdentity | None = None,
    design: ControlledInputDesign | None = None,
) -> LocalizationGrid:
    """Capture one SAE code at each controlled input state in bitmask order.

    The caller must construct the independent binary-factor design and align
    the same semantic activation position across prompts. No feature patches
    are performed; input interventions are represented by the supplied prompts.
    """
    if design is not None:
        if prompts_by_state is not None or n_factors is not None:
            raise ValueError("provide design or legacy prompts_by_state/n_factors, not both")
        prompts = tuple(state.prompt for state in design.states)
        n_factors = design.n_factors
    else:
        prompts = _ordered_states(prompts_by_state, n_factors, label="prompts_by_state")
    codes, activations = [], []
    if identity is not None and (identity.hook_name != site.hook_name
                                 or identity.layer != site.layer):
        raise ValueError("SAE identity site differs from requested site")
    for index, prompt in enumerate(prompts):
        selected_site = (replace(site, position=design.states[index].position)
                         if design is not None else site)
        snapshot = capture_features(model, sae, site=selected_site, prompt=prompt)
        codes.append(snapshot.codes.detach().cpu().numpy().astype(np.float64, copy=True))
        activations.append(snapshot.activation.detach().cpu().numpy().astype(np.float64, copy=True))
    return LocalizationGrid(
        codes=np.stack(codes), activations=np.stack(activations),
        n_factors=n_factors,
        state_sample_ids=(tuple(state.sample_id for state in design.states)
                          if design is not None else tuple(f"state-{i}" for i in range(len(prompts)))),
        source="saebase.capture_features", site=site, identity=identity,
        design_key=design.key if design is not None else None,
    )


def localization_grid_from_code_store(
    code_store_path: str | Path,
    *,
    sample_ids_by_state: Mapping[int, str] | None = None,
    n_factors: int | None = None,
    site: ActivationSite | None = None,
    identity: SAEIdentity | None = None,
    manifest: ControlledInputManifest | str | Path | None = None,
) -> LocalizationGrid:
    """Read a complete input grid from SAEBase's integrity-checked code cache.

    Each sample must select exactly one activation position. The paired parent
    activation store is checked against each code's source-activation digest.
    """
    from saebase.store import ShardedReader

    controlled = None
    if manifest is not None:
        if sample_ids_by_state is not None or n_factors is not None:
            raise ValueError("provide a controlled design manifest or legacy state IDs, not both")
        controlled = load_controlled_input_manifest(manifest)
        n_factors = controlled.n_factors
        sample_ids = tuple(controlled.sample_ids_by_state[i] for i in range(1 << n_factors))
    else:
        sample_ids = _ordered_states(sample_ids_by_state, n_factors, label="sample_ids_by_state")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids_by_state must use distinct samples for all states")
    path = Path(code_store_path)
    codes_reader = ShardedReader(path, verify_shards=True)
    if codes_reader.kind != "codes":
        raise ValueError("code_store_path must refer to an SAE code store")
    activation_reader = ShardedReader(path.parent.parent, verify_shards=True)
    if activation_reader.kind != "activations":
        raise ValueError("code store must have its SAEBase parent activation store")
    code_identity = codes_reader.manifest["identity"]
    activation_identity = activation_reader.manifest["identity"]
    if controlled is not None and controlled.activation_identity_key != activation_reader.manifest["identity_key"]:
        raise ValueError("controlled input manifest belongs to a different activation cache")
    if code_identity["activation_key"] != activation_reader.manifest["identity_key"]:
        raise ValueError("SAE code cache no longer matches its parent activation store")
    if site is not None and (site.hook_name != activation_identity["hook_name"]
                             or site.layer != activation_identity["layer"]):
        raise ValueError("requested site differs from cached activation site")
    if identity is not None:
        cached_sae = code_identity["sae"]
        if (identity.sae_key != cached_sae["sae_key"]
                or identity.d_in != cached_sae["d_in"]
                or identity.d_sae != cached_sae["d_sae"]):
            raise ValueError("SAE identity differs from cached SAE codes")
        if (identity.hook_name != activation_identity["hook_name"]
                or identity.layer != activation_identity["layer"]):
            raise ValueError("SAE identity site differs from cached activation site")
    rows = []
    from saebase.store import fingerprint
    for mask, sample_id in enumerate(sample_ids):
        if not codes_reader.has(sample_id) or not activation_reader.has(sample_id):
            raise ValueError(f"missing cached input state: {sample_id}")
        code_sample = codes_reader.get(sample_id)
        activation_sample = activation_reader.get(sample_id)
        if len(code_sample.positions) != 1 or len(activation_sample.positions) != 1:
            raise ValueError("Localize requires exactly one captured token position per state")
        if code_sample.positions != activation_sample.positions:
            raise ValueError("code and activation positions differ")
        if replace(code_sample.record, source_activation_sha256=None).key != activation_sample.record.key:
            raise ValueError("SAE codes and parent activations have different sample metadata")
        if controlled is not None:
            expected = controlled.states[mask]
            site_digest = fingerprint({
                "token_ids": list(code_sample.record.token_ids),
                "positions": list(code_sample.positions),
            })
            if (site_digest != expected["token_site_sha256"]
                    or code_sample.positions[0] != expected["position"]):
                raise ValueError("controlled input manifest token or position mismatch")
        sha = hashlib.sha256(np.asarray(activation_sample.values, dtype=np.float32).tobytes(order="C")).hexdigest()
        if code_sample.record.source_activation_sha256 != sha:
            raise ValueError("cached SAE code source-activation checksum mismatch")
        rows.append(np.array(code_sample.values[0], dtype=np.float64, copy=True))
    return LocalizationGrid(
        codes=np.stack(rows), n_factors=n_factors, state_sample_ids=sample_ids,
        source="saebase.code_store", site=site, identity=identity,
        design_key=controlled.design_key if controlled is not None else None,
    )


def _original_localize(paper: Any | None) -> Any:
    if paper is None:
        try:
            paper = importlib.import_module("sae_localize")
        except ImportError as exc:
            raise ImportError(
                "Install the original Localize package from "
                "https://github.com/OpenSAE/SAE-Input-Localization (pip install -e .)"
            ) from exc
    if not callable(getattr(paper, "exact_scope_curve", None)):
        raise TypeError("original Localize package must expose exact_scope_curve()")
    return paper


def evaluate_localization(
    grid: LocalizationGrid,
    *,
    paper: Any | None = None,
    relative_omission: float = 0.1,
    pairs_per_scope: int | None = None,
    seed: int = 0,
) -> LocalizationResult:
    """Delegate exact (and optional plug-in paired) curves to original Localize.

    Full-grid exact optimization uses the uniform Boolean product distribution.
    The finite-pair estimate is exploratory, NOT a post-selection certificate.
    """
    if not isinstance(grid, LocalizationGrid):
        raise TypeError("grid must be a LocalizationGrid")
    if (isinstance(relative_omission, bool) or not np.isscalar(relative_omission)
            or not np.isfinite(relative_omission) or not 0 <= relative_omission <= 1):
        raise ValueError("relative_omission must be between 0 and 1")
    if pairs_per_scope is not None and (type(pairs_per_scope) is not int or pairs_per_scope < 1):
        raise ValueError("pairs_per_scope must be a positive integer or None")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    original = _original_localize(paper)
    z = np.asarray(grid.codes, dtype=np.float64)
    native = original.exact_scope_curve(z, n_factors=grid.n_factors)
    curve = np.asarray(native.curve, dtype=np.float64).copy()
    expected_length = grid.n_factors * grid.n_features + 1
    if curve.shape != (expected_length,) or not np.isfinite(curve).all():
        raise ValueError("original Localize estimator returned an invalid scope curve")
    variance = float(native.total_variance)
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("original Localize estimator returned invalid total variance")
    tolerance = float(relative_omission) * variance
    budget = int(native.budget(tolerance))
    if not 0 <= budget < expected_length:
        raise ValueError("original Localize estimator returned invalid budget")
    assignment = np.asarray(native.assignment(budget)).copy()
    if (assignment.shape != (grid.n_features,)
            or not np.issubdtype(assignment.dtype, np.integer)
            or np.any(assignment < 0) or np.any(assignment >= 1 << grid.n_factors)):
        raise ValueError("original Localize estimator returned invalid feature scopes")
    estimated = None
    if pairs_per_scope is not None:
        if not callable(getattr(original, "estimated_scope_curve", None)):
            raise TypeError("original Localize package must expose estimated_scope_curve()")
        estimated = np.asarray(
            original.estimated_scope_curve(z, pairs_per_scope=pairs_per_scope, seed=seed),
            dtype=np.float64,
        ).copy()
        if estimated.shape != curve.shape or not np.isfinite(estimated).all():
            raise ValueError("original Localize paired estimator returned an invalid curve")
    source = getattr(original.exact_scope_curve, "__code__", None)
    filename = source.co_filename if source is not None else None
    digest = (hashlib.sha256(Path(filename).read_bytes()).hexdigest()
              if filename and Path(filename).is_file() else None)
    return LocalizationResult(
        curve=curve, assignment=assignment, budget=budget, total_variance=variance,
        relative_omission=float(relative_omission), grid=grid,
        estimator_version=getattr(original, "__version__", None),
        exact_source_sha256=digest, estimated_curve=estimated,
        pairs_per_scope=pairs_per_scope, seed=seed if pairs_per_scope is not None else None,
    )
