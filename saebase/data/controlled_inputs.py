"""Complete Boolean input designs and provenance for SAE activation experiments.

Controlled input states are external model inputs, not SAE feature edits.
Existing SAEBase SampleRecord and activation/code storage schemas are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from saebase.store import (
    ActivationWriter, CollectionStats, SampleRecord, TokenSelection,
    collect_activations, fingerprint, resolve_positions,
)
from saebase.api import ActivationSite, ModelAdapter

MANIFEST_SCHEMA = "saebase.controlled_inputs.v1"
MAX_FACTORS = 12


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _state_mapping(items: Mapping[int, Any], n: int, label: str) -> tuple[Any, ...]:
    if not isinstance(items, Mapping):
        raise TypeError(f"{label} must map integer state masks to values")
    count = 1 << n
    if (len(items) != count or any(type(key) is not int for key in items)
            or set(items) != set(range(count))):
        raise ValueError(f"{label} must cover each state mask 0..{count - 1} exactly once")
    return tuple(items[i] for i in range(count))


@dataclass(frozen=True)
class BinaryFactor:
    name: str
    levels: tuple[str, str] = ("0", "1")

    def __post_init__(self) -> None:
        _name(self.name, "factor name")
        if (not isinstance(self.levels, tuple) or len(self.levels) != 2
                or any(not isinstance(v, str) or not v.strip() for v in self.levels)
                or self.levels[0] == self.levels[1]):
            raise ValueError("binary factor levels must be two distinct nonempty strings")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "levels": list(self.levels)}


@dataclass(frozen=True)
class ControlledInputState:
    mask: int
    prompt: str
    sample_id: str
    position: int = -1

    def __post_init__(self) -> None:
        if type(self.mask) is not int or self.mask < 0:
            raise ValueError("state mask must be a nonnegative integer")
        _name(self.prompt, "input prompt")
        _name(self.sample_id, "sample_id")
        if type(self.position) is not int:
            raise ValueError("token position must be an integer")


@dataclass(frozen=True)
class ControlledInputDesign:
    """Exactly one prompt for every combination of independent binary factors.

    Bit k of mask i selects the corresponding factor's 0/1 level. The caller
    is responsible for the *semantic* validity of these interventions.
    """

    factors: tuple[BinaryFactor, ...]
    states: tuple[ControlledInputState, ...]
    name: str = "controlled-inputs"

    def __post_init__(self) -> None:
        _name(self.name, "design name")
        if (not isinstance(self.factors, tuple) or not self.factors
                or len(self.factors) > MAX_FACTORS
                or any(not isinstance(f, BinaryFactor) for f in self.factors)):
            raise ValueError(f"factors must contain 1..{MAX_FACTORS} BinaryFactor items")
        names = [f.name for f in self.factors]
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique")
        count = 1 << len(self.factors)
        if (not isinstance(self.states, tuple) or len(self.states) != count
                or any(not isinstance(s, ControlledInputState) for s in self.states)
                or tuple(s.mask for s in self.states) != tuple(range(count))):
            raise ValueError("states must contain the entire Boolean grid in bitmask order")
        if len({s.prompt for s in self.states}) != count:
            raise ValueError("different input configurations must have distinct prompts")
        if len({s.sample_id for s in self.states}) != count:
            raise ValueError("each input configuration must have a distinct sample ID")

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def key(self) -> str:
        """Design identity changes when factors or the input-state map changes."""
        return fingerprint({
            "schema": MANIFEST_SCHEMA,
            "name": self.name,
            "factors": [f.to_dict() for f in self.factors],
            "states": [{"mask": s.mask, "sample_id": s.sample_id,
                        "prompt_sha256": hashlib.sha256(s.prompt.encode("utf-8")).hexdigest(),
                        "position": s.position} for s in self.states],
        })

    @property
    def prompt_group_id(self) -> str:
        """Invariant under state reindexing of the same complete input population."""
        return "controlled-" + fingerprint(sorted(s.sample_id for s in self.states))[:20]

    @property
    def prompts_by_state(self) -> dict[int, str]:
        return {s.mask: s.prompt for s in self.states}

    @property
    def sample_ids_by_state(self) -> dict[int, str]:
        return {s.mask: s.sample_id for s in self.states}

    @classmethod
    def from_prompts(
        cls, *, factors: Sequence[BinaryFactor], prompts_by_state: Mapping[int, str],
        name: str = "controlled-inputs",
        positions_by_state: Mapping[int, int] | None = None,
        sample_ids_by_state: Mapping[int, str] | None = None,
    ) -> ControlledInputDesign:
        factors = tuple(factors)
        if not factors or len(factors) > MAX_FACTORS or any(not isinstance(f, BinaryFactor) for f in factors):
            raise ValueError(f"factors must contain 1..{MAX_FACTORS} BinaryFactor items")
        prompts = _state_mapping(prompts_by_state, len(factors), "prompts_by_state")
        positions = (_state_mapping(positions_by_state, len(factors), "positions_by_state")
                     if positions_by_state is not None else (-1,) * len(prompts))
        if sample_ids_by_state is None:
            ids = tuple(
                "controlled-" + fingerprint({"prompt": prompt, "position": pos})[:24]
                for prompt, pos in zip(prompts, positions)
            )
        else:
            ids = _state_mapping(sample_ids_by_state, len(factors), "sample_ids_by_state")
        states = tuple(
            ControlledInputState(i, prompt, sid, position)
            for i, (prompt, sid, position) in enumerate(zip(prompts, ids, positions))
        )
        return cls(factors=factors, states=states, name=name)

    @classmethod
    def from_renderer(
        cls, *, factors: Sequence[BinaryFactor], render: Callable[[dict[str, str]], str],
        name: str = "controlled-inputs", position: int = -1,
    ) -> ControlledInputDesign:
        factors = tuple(factors)
        if not factors or len(factors) > MAX_FACTORS or any(not isinstance(f, BinaryFactor) for f in factors):
            raise ValueError(f"factors must contain 1..{MAX_FACTORS} BinaryFactor items")
        if not callable(render):
            raise TypeError("render must be a callable")
        if type(position) is not int:
            raise ValueError("position must be an integer")
        prompts = {
            mask: render({factor.name: factor.levels[(mask >> bit) & 1]
                          for bit, factor in enumerate(factors)})
            for mask in range(1 << len(factors))
        }
        return cls.from_prompts(
            factors=factors, prompts_by_state=prompts, name=name,
            positions_by_state={mask: position for mask in prompts},
        )

    def reindex(self, new_to_old: Mapping[int, int], *, name: str | None = None) -> ControlledInputDesign:
        """Reassign the *same* input samples to a bijective input coordinate map.

        Input activations are reused. Reindexing changes the interpretation of
        factors; it does not perform another model-input intervention.
        """
        old_masks = _state_mapping(new_to_old, self.n_factors, "new_to_old")
        if (any(type(i) is not int for i in old_masks)
                or set(old_masks) != set(range(self.n_states))):
            raise ValueError("new_to_old must be a permutation of all input state masks")
        return ControlledInputDesign(
            factors=self.factors,
            states=tuple(
                ControlledInputState(new, self.states[old].prompt,
                                     self.states[old].sample_id,
                                     self.states[old].position)
                for new, old in enumerate(old_masks)
            ),
            name=self.name if name is None else name,
        )

    def sample_records(
        self, model: ModelAdapter, *, split: str = "evaluation",
        prompt_group_id: str | None = None,
    ) -> tuple[SampleRecord, ...]:
        """Tokenize and validate every state; select exactly one model token."""
        _name(split, "split")
        group = self.prompt_group_id if prompt_group_id is None else _name(prompt_group_id, "prompt_group_id")
        records: list[SampleRecord] = []
        seen_tokens: set[tuple[int, ...]] = set()
        for state in self.states:
            ids = model.tokens(prompt=state.prompt).detach().cpu().reshape(-1)
            tokens = tuple(int(v) for v in ids.tolist())
            if tokens in seen_tokens:
                raise ValueError("different controlled states produced identical token IDs")
            seen_tokens.add(tokens)
            pos = state.position
            if pos < -len(tokens) or pos >= len(tokens):
                raise ValueError(f"state {state.mask} token position outside sequence")
            selection = TokenSelection(mode="indices", indices=(pos % len(tokens),))
            records.append(SampleRecord(
                sample_id=state.sample_id, token_ids=tokens, selection=selection,
                split=split, prompt_group_id=group, variant_role="controlled",
            ))
        return tuple(records)


@dataclass(frozen=True)
class ControlledInputManifest:
    """Immutable, text-free mapping from input states to cached token samples."""

    name: str
    design_key: str
    factors: tuple[BinaryFactor, ...]
    states: tuple[dict[str, Any], ...]
    activation_identity_key: str

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def sample_ids_by_state(self) -> dict[int, str]:
        return {int(row["mask"]): str(row["sample_id"]) for row in self.states}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": MANIFEST_SCHEMA,
            "name": self.name, "design_key": self.design_key,
            "distribution": "uniform_binary_product",
            "factors": [f.to_dict() for f in self.factors],
            "activation_identity_key": self.activation_identity_key,
            "states": [dict(s) for s in self.states],
        }
        return {**payload, "manifest_sha256": fingerprint(payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlledInputManifest:
        if not isinstance(data, dict):
            raise ValueError("controlled input manifest must be an object")
        keys = {"schema", "name", "design_key", "distribution", "factors",
                "activation_identity_key", "states", "manifest_sha256"}
        if set(data) != keys or data.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("invalid controlled input manifest schema")
        if data.get("distribution") != "uniform_binary_product":
            raise ValueError("controlled design requires a uniform binary product grid")
        digest = data["manifest_sha256"]
        payload = {k: v for k, v in data.items() if k != "manifest_sha256"}
        if type(digest) is not str or fingerprint(payload) != digest:
            raise ValueError("controlled input manifest checksum mismatch")
        factors = tuple(BinaryFactor(f["name"], tuple(f["levels"])) for f in data["factors"])
        if not factors or len(factors) > MAX_FACTORS or len({f.name for f in factors}) != len(factors):
            raise ValueError("invalid controlled input factors")
        _name(data["name"], "design name")
        if any(type(data[k]) is not str or len(data[k]) != 64
               or any(ch not in "0123456789abcdef" for ch in data[k])
               for k in ("design_key", "activation_identity_key")):
            raise ValueError("invalid design or activation identity digest")
        states = data["states"]
        if not isinstance(states, list) or len(states) != 1 << len(factors):
            raise ValueError("manifest requires a complete input state grid")
        sample_ids = []
        checked_rows = []
        for mask, row in enumerate(states):
            if not isinstance(row, dict) or set(row) != {"mask", "levels", "sample_id", "token_site_sha256", "position"}:
                raise ValueError("invalid controlled input state row")
            if row["mask"] != mask or type(row["mask"]) is not int:
                raise ValueError("controlled input states must use bitmask order")
            if row["levels"] != [(mask >> i) & 1 for i in range(len(factors))]:
                raise ValueError("controlled input factor levels mismatch state mask")
            _name(row["sample_id"], "sample_id")
            if (type(row["position"]) is not int or row["position"] < 0
                    or type(row["token_site_sha256"]) is not str
                    or len(row["token_site_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in row["token_site_sha256"])):
                raise ValueError("invalid token provenance for controlled state")
            sample_ids.append(row["sample_id"])
            checked_rows.append(dict(row))
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("manifest contains duplicate sample IDs")
        return cls(data["name"], data["design_key"], factors, tuple(checked_rows),
                   data["activation_identity_key"])


def _token_site_digest(record: SampleRecord) -> str:
    return fingerprint({"token_ids": list(record.token_ids),
                        "positions": list(resolve_positions(record))})


def make_controlled_input_manifest(
    design: ControlledInputDesign,
    records: Sequence[SampleRecord],
    *,
    activation_identity_key: str,
) -> ControlledInputManifest:
    if len(records) != design.n_states:
        raise ValueError("record count must match full controlled input grid")
    rows = []
    for state, record in zip(design.states, records):
        if state.sample_id != record.sample_id:
            raise ValueError("record order or sample identity disagrees with design")
        positions = resolve_positions(record)
        if len(positions) != 1:
            raise ValueError("controlled input state must select exactly one token")
        rows.append({
            "mask": state.mask,
            "levels": [(state.mask >> i) & 1 for i in range(design.n_factors)],
            "sample_id": state.sample_id,
            "token_site_sha256": _token_site_digest(record),
            "position": positions[0],
        })
    manifest = ControlledInputManifest(
        name=design.name, design_key=design.key, factors=design.factors,
        states=tuple(rows), activation_identity_key=activation_identity_key,
    )
    return ControlledInputManifest.from_dict(manifest.to_dict())


def save_controlled_input_manifest(path: str | Path, manifest: ControlledInputManifest) -> Path:
    """Atomically create a sidecar; reject silent overwrite with different data."""
    path = Path(path)
    serialized = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_controlled_input_manifest(path).to_dict() != manifest.to_dict():
            raise ValueError(f"controlled input manifest already exists with different contents: {path}")
        return path
    fd, temp = tempfile.mkstemp(prefix=".controlled-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if load_controlled_input_manifest(path).to_dict() != manifest.to_dict():
                raise ValueError(f"controlled input manifest already exists with different contents: {path}")
        else:
            os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def load_controlled_input_manifest(source: str | Path | ControlledInputManifest) -> ControlledInputManifest:
    if isinstance(source, ControlledInputManifest):
        return ControlledInputManifest.from_dict(source.to_dict())
    return ControlledInputManifest.from_dict(json.loads(Path(source).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ControlledInputCapture:
    records: tuple[SampleRecord, ...]
    stats: CollectionStats
    manifest_path: Path
    manifest: ControlledInputManifest


def collect_controlled_input_activations(
    model: ModelAdapter, design: ControlledInputDesign,
    writer: ActivationWriter, *, site: ActivationSite,
    manifest_path: str | Path | None = None,
) -> ControlledInputCapture:
    """Collect/reuse exact-input caches, then write a versioned design sidecar."""
    if (writer.activation_identity.hook_name != site.hook_name
            or writer.activation_identity.layer != site.layer):
        raise ValueError("activation store and requested site differ")
    records = design.sample_records(model)
    stats = collect_activations(model, records, writer, site=site)
    manifest = make_controlled_input_manifest(
        design, records, activation_identity_key=writer.activation_identity.key
    )
    destination = (Path(writer.path) / "controlled_inputs" / f"{design.key}.json"
                   if manifest_path is None else Path(manifest_path))
    save_controlled_input_manifest(destination, manifest)
    return ControlledInputCapture(records, stats, destination, manifest)
