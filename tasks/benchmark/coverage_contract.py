"""Stage B/C model-scale, SAE-architecture and layer coverage contracts.

The catalog is a plan, not verification evidence. Real CPU checkpoint status
is represented by separate checksum-bearing output artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from saebase.api.sae_profiles import get_profile

SCHEMA = "saebase.coverage.stage_bc.v1"
BANDS = (("early", 0.25), ("middle", 0.50),
         ("middle_late", 0.65), ("late", 0.85))


@dataclass(frozen=True)
class CoverageEntry:
    model_key: str
    sae_key: str
    n_layers: int
    size: str
    architecture: str
    selected_layers: tuple[int, ...]
    cpu_probe_layers: tuple[int, ...]
    access: str
    tier: str
    model_pair_cpu_candidate: bool

    def __post_init__(self) -> None:
        p = get_profile(self.sae_key)
        if self.model_key not in p.allowed_model_keys():
            raise ValueError(f"{self.sae_key}: source model mismatch")
        if not isinstance(self.n_layers, int) or self.n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        if not self.selected_layers or len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError("selected_layers must be nonempty and unique")
        if any(type(i) is not int or i < 0 or i >= self.n_layers or i not in p.layers
               for i in self.selected_layers):
            raise ValueError(f"{self.sae_key}: selected layers disagree with registry")
        if (len(set(self.cpu_probe_layers)) != len(self.cpu_probe_layers)
                or not set(self.cpu_probe_layers).issubset(self.selected_layers)):
            raise ValueError("CPU probe layers must be a unique subset")
        if self.tier not in ("core", "architecture"):
            raise ValueError("unknown coverage tier")
        if self.model_pair_cpu_candidate and (
            self.access != "open" or self.tier != "core"
        ):
            raise ValueError("CPU pair target must be an open core model")

    def source(self, layer: int) -> dict[str, Any]:
        if layer not in self.selected_layers:
            raise ValueError(f"unregistered layer {layer}")
        p = get_profile(self.sae_key)
        release, source_id, converter = p.sae_release, p.sae_id(layer), None
        if p.loader == "sae_lens":
            from saebase.api.sae_loading import _phase_a_sae_lens_source
            release, source_id, converter = _phase_a_sae_lens_source(p, layer)
        elif p.loader in ("sparsify_topk", "dictionary_learning", "eleutherai_topk_mlp"):
            source_id = p.sae_folder(layer) or source_id
        return {
            "model_key": self.model_key, "sae_key": self.sae_key,
            "size": self.size, "tier": self.tier,
            "layer": layer, "n_layers": self.n_layers,
            "depth_fraction": round((layer + 1) / self.n_layers, 5),
            "architecture": self.architecture,
            "hook_name": p.hook_name(layer),
            "loader": p.loader, "repo_id": p.repo_id or release,
            "source_release": release,
            "source_file_id": source_id,
            "converter": converter,
            "access": self.access,
            "planned_cpu_sae_probe": layer in self.cpu_probe_layers,
            "planned_cpu_model_pair_probe": self.model_pair_cpu_candidate and
                                             layer == self.cpu_probe_layers[0],
            "verification": "registered_only",
        }

    def depth_bands(self) -> dict[str, dict[str, Any]]:
        """Greedy distinct selection of the closest available actual layers."""
        available = set(self.selected_layers)
        bands = {}
        for band, fraction in BANDS:
            if not available:
                bands[band] = {"layer": None, "target_fraction": fraction}
                continue
            # Zero-indexed layer depth; deterministic ties prefer lower layers.
            closest = min(
                available, key=lambda l: (abs((l + 1) / self.n_layers - fraction), l)
            )
            available.remove(closest)
            bands[band] = {
                "layer": closest, "target_fraction": fraction,
                "actual_fraction": round((closest + 1) / self.n_layers, 5),
            }
        return bands


@dataclass(frozen=True)
class CoverageCatalog:
    entries: tuple[CoverageEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("empty coverage catalog")
        keys = [e.sae_key for e in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate SAE entries")
        core_models = [e.model_key for e in self.entries if e.tier == "core"]
        if len(set(core_models)) != len(core_models):
            raise ValueError("duplicate core model in coverage catalog")

    def find(self, sae_key: str) -> CoverageEntry:
        for item in self.entries:
            if item.sae_key == sae_key:
                return item
        raise KeyError(f"SAE not in Stage B/C catalog: {sae_key}")

    def plan(self) -> dict[str, Any]:
        entries = []
        for entry in self.entries:
            entries.append({
                "model_key": entry.model_key, "sae_key": entry.sae_key,
                "architecture": entry.architecture, "tier": entry.tier,
                "size": entry.size, "access": entry.access,
                "bands": entry.depth_bands(),
                "checkpoints": [entry.source(l) for l in entry.selected_layers],
            })
        return {
            "schema": SCHEMA,
            "kind": "registry_plan_not_real_checkpoint_evidence",
            "core_models": len([e for e in self.entries if e.tier == "core"]),
            "extra_architectures": len([e for e in self.entries if e.tier == "architecture"]),
            "planned_checkpoints": sum(len(e.selected_layers) for e in self.entries),
            "planned_cpu_sae_probes": sum(len(e.cpu_probe_layers) for e in self.entries),
            "planned_cpu_model_pair_probes": sum(e.model_pair_cpu_candidate for e in self.entries),
            "entries": entries,
        }


def load_coverage(path: str | Path) -> CoverageCatalog:
    src = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(src, dict) or src.get("schema") != SCHEMA:
        raise ValueError("invalid coverage catalog schema")
    valid_fields = set(CoverageEntry.__dataclass_fields__)
    entries = []
    for x in src.get("targets", []):
        if not isinstance(x, dict) or set(x) != valid_fields:
            raise ValueError("catalog entry fields differ from Stage B/C contract")
        items = dict(x)
        items["selected_layers"] = tuple(items["selected_layers"])
        items["cpu_probe_layers"] = tuple(items["cpu_probe_layers"])
        entries.append(CoverageEntry(**items))
    return CoverageCatalog(tuple(entries))


def verify_cpu_sae_evidence(catalog: CoverageCatalog, record: dict[str, Any]) -> dict[str, Any]:
    """Check a run's *claims* against source identity and real weight proof fields.

    This validates the published record, but independent cryptographic trust
    also requires the GitHub Actions source commit and artifact provenance.
    """
    import math
    import re

    if not isinstance(record, dict):
        raise ValueError("CPU SAE evidence must be JSON object")
    if record.get("schema") != "saebase.coverage.cpu_sae_acceptance.v1":
        raise ValueError("unrecognized CPU SAE evidence schema")
    sae_key, layer = record.get("sae_key"), record.get("layer")
    if not isinstance(sae_key, str) or type(layer) is not int:
        raise ValueError("CPU SAE evidence identity missing")
    entry = catalog.find(sae_key)
    if layer not in entry.cpu_probe_layers:
        raise ValueError("SAE evidence layer was not selected for CPU verification")
    if record.get("status") != "sae_checkpoint_verified_only":
        return {"sae_key":sae_key,"layer":layer,"status":"blocked"}
    source=entry.source(layer)
    if record.get("model_sae_pair_verified") is not False or record.get("model_loaded") is not False:
        raise ValueError("SAE-only evidence cannot assert model-pair verification")
    if (record.get("model_key")!=entry.model_key or
        record.get("hook_name")!=source["hook_name"] or
        record.get("source")!=source):
        raise ValueError("SAE weight evidence source identity mismatch")
    digest=record.get("sae_state_sha256")
    if not isinstance(digest,str) or re.fullmatch(r"[0-9a-f]{64}",digest) is None:
        raise ValueError("successful SAE evidence must contain full checkpoint SHA256")
    d_in,d_sae=record.get("d_in"),record.get("d_sae")
    if (type(d_in) is not int or type(d_sae) is not int or
        not 0<d_in<d_sae or d_sae>500000):
        raise ValueError("SAE evidence dimensions invalid")
    values=record.get("synthetic_probe_l0")
    mse=record.get("synthetic_probe_reconstruction_mse")
    if (not isinstance(values,list) or len(values)!=2 or
        any(type(x) is not int or x<0 or x>d_sae for x in values) or
        type(mse) not in (int,float) or not math.isfinite(mse) or mse<0):
        raise ValueError("SAE evidence lacks finite encode/decode probe")
    return {"sae_key":sae_key,"layer":layer,"status":"sae_checkpoint_verified_only",
            "sae_state_sha256":digest,"d_in":d_in,"d_sae":d_sae}
