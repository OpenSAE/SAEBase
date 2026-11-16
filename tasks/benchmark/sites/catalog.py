"""Purpose: Declare selected model SAE sites, plan their order and inspect checkpoint coverage and sources.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from saebase.api.sae_profiles import get_profile

SCHEMA = "saebase.coverage.stage_bc.v1"

BANDS = (("early", 0.25), ("middle", 0.50), ("middle_late", 0.65), ("late", 0.85))


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
        if any(
            type(i) is not int or i < 0 or i >= self.n_layers or i not in p.layers
            for i in self.selected_layers
        ):
            raise ValueError(f"{self.sae_key}: selected layers disagree with registry")
        if len(set(self.cpu_probe_layers)) != len(self.cpu_probe_layers) or not set(
            self.cpu_probe_layers
        ).issubset(self.selected_layers):
            raise ValueError("CPU probe layers must be a unique subset")
        if self.tier not in ("core", "architecture"):
            raise ValueError("unknown coverage tier")
        if self.model_pair_cpu_candidate and (self.access != "open" or self.tier != "core"):
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
            "model_key": self.model_key,
            "sae_key": self.sae_key,
            "size": self.size,
            "tier": self.tier,
            "layer": layer,
            "n_layers": self.n_layers,
            "depth_fraction": round((layer + 1) / self.n_layers, 5),
            "architecture": self.architecture,
            "hook_name": p.hook_name(layer),
            "loader": p.loader,
            "repo_id": p.repo_id or release,
            "source_release": release,
            "source_file_id": source_id,
            "converter": converter,
            "access": self.access,
            "planned_cpu_sae_probe": layer in self.cpu_probe_layers,
            "planned_cpu_model_pair_probe": self.model_pair_cpu_candidate
            and layer == self.cpu_probe_layers[0],
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
                available, key=lambda layer: (abs((layer + 1) / self.n_layers - fraction), layer)
            )
            available.remove(closest)
            bands[band] = {
                "layer": closest,
                "target_fraction": fraction,
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
            entries.append(
                {
                    "model_key": entry.model_key,
                    "sae_key": entry.sae_key,
                    "architecture": entry.architecture,
                    "tier": entry.tier,
                    "size": entry.size,
                    "access": entry.access,
                    "bands": entry.depth_bands(),
                    "checkpoints": [entry.source(layer) for layer in entry.selected_layers],
                }
            )
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
        return {"sae_key": sae_key, "layer": layer, "status": "blocked"}
    source = entry.source(layer)
    if (
        record.get("model_sae_pair_verified") is not False
        or record.get("model_loaded") is not False
    ):
        raise ValueError("SAE-only evidence cannot assert model-pair verification")
    if (
        record.get("model_key") != entry.model_key
        or record.get("hook_name") != source["hook_name"]
        or record.get("source") != source
    ):
        raise ValueError("SAE weight evidence source identity mismatch")
    digest = record.get("sae_state_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("successful SAE evidence must contain full checkpoint SHA256")
    d_in, d_sae = record.get("d_in"), record.get("d_sae")
    if type(d_in) is not int or type(d_sae) is not int or not 0 < d_in < d_sae or d_sae > 500000:
        raise ValueError("SAE evidence dimensions invalid")
    values = record.get("synthetic_probe_l0")
    mse = record.get("synthetic_probe_reconstruction_mse")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(type(x) is not int or x < 0 or x > d_sae for x in values)
        or type(mse) not in (int, float)
        or not math.isfinite(mse)
        or mse < 0
    ):
        raise ValueError("SAE evidence lacks finite encode/decode probe")
    return {
        "sae_key": sae_key,
        "layer": layer,
        "status": "sae_checkpoint_verified_only",
        "sae_state_sha256": digest,
        "d_in": d_in,
        "d_sae": d_sae,
    }


CATALOG = Path(__file__).resolve().parents[3] / "conf/coverage/stage_bc.json"

CPU_RESULT_SCHEMA = "saebase.coverage.cpu_matrix_result.v2"


def _records(folder: Path, catalog) -> dict[tuple[str, int], dict[str, Any]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"missing CPU artifact directory: {folder}")
    files = sorted(folder.rglob("*.json"))
    if not files:
        raise ValueError(f"no evidence JSON artifacts found in {folder}")
    entries = {}
    for path in files:
        raw = path.read_bytes()
        data = json.loads(raw)
        accepted = verify_cpu_sae_evidence(catalog, data)
        key = (accepted["sae_key"], accepted["layer"])
        if key in entries:
            raise ValueError(f"duplicated evidence for {key}")
        entries[key] = {
            **accepted,
            "evidence_file": path.name,
            "artifact_json_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return entries


def aggregate(
    source_dir: Path,
    catalog_path: Path = CATALOG,
    replacements_dir: Path | None = None,
) -> dict[str, Any]:
    catalog = load_coverage(catalog_path)
    expected = {
        (item.sae_key, layer) for item in catalog.entries for layer in item.cpu_probe_layers
    }
    base = _records(source_dir, catalog)
    if set(base) - expected:
        raise ValueError("baseline artifacts include undeclared SAE checkpoint")
    if replacements_dir is not None:
        repaired = _records(replacements_dir, catalog)
        if set(repaired) - expected:
            raise ValueError("replacement artifacts include undeclared checkpoint")
        for key, replacement in repaired.items():
            existing = base.get(key)
            if existing and existing["status"] == "sae_checkpoint_verified_only":
                if existing["sae_state_sha256"] != replacement.get("sae_state_sha256"):
                    raise ValueError("cannot silently replace a previously verified SAE hash")
                continue
            if replacement["status"] != "sae_checkpoint_verified_only":
                raise ValueError("replacement must be a successful genuine checkpoint artifact")
            base[key] = {**replacement, "repaired_after_block": existing is not None}
    missing = sorted(expected - set(base))
    passed = sorted(x for x, v in base.items() if v["status"] == "sae_checkpoint_verified_only")
    blocked = sorted(x for x, v in base.items() if v["status"] != "sae_checkpoint_verified_only")
    results = [{**base[key]} for key in sorted(base)]
    return {
        "schema": CPU_RESULT_SCHEMA,
        "kind": "original_SAE_weights_only_not_model_SAE_pair_verification",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "expected": len(expected),
        "with_artifact": len(base),
        "sae_only_verified": len(passed),
        "failed_or_blocked": len(blocked),
        "missing": len(missing),
        "blocked_checkpoints": [f"{k[0]} L{k[1]}" for k in blocked],
        "missing_checkpoints": [f"{k[0]} L{k[1]}" for k in missing],
        "entries": results,
        "all_checkpoints_verified": not missing and not blocked,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None):
    """Inspect selected coverage or aggregate existing checkpoint observations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "aggregate", "sources"])
    parser.add_argument("--catalog", "--config", type=Path, default=CATALOG)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--replacements-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("coverage_result.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "plan":
        plan = load_coverage(args.catalog).plan()
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    if args.action == "sources":
        result = check_sources(args.catalog)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["failed"] == 0 and result["unresolved"] == 0 else 1
    if args.source_dir is None:
        parser.error("aggregate requires --source-dir")
    result = aggregate(args.source_dir, args.catalog, args.replacements_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "result": str(args.output),
                "missing": result["missing_checkpoints"],
                "blocked": result["blocked_checkpoints"],
            },
            indent=2,
        )
    )
    return 0 if result["all_checkpoints_verified"] else 1


def probe_file_path(entry: CoverageEntry, layer: int) -> str | None:
    p = entry.source(layer)
    key = entry.sae_key
    if p["loader"] == "qwen_resid_topk":
        return p["source_file_id"]
    if p["loader"] in ("eleutherai_topk_mlp", "sparsify_topk"):
        return p["source_file_id"].rstrip("/") + "/sae.safetensors"
    if p["loader"] == "dictionary_learning":
        return p["source_file_id"].rstrip("/") + "/ae.pt"
    if key.startswith("gemma3_4b_") or key in {
        "gemma270m_scope_resid",
        "gemma1b_scope_resid",
    }:
        return p["source_file_id"].rstrip("/") + "/params.safetensors"
    if key == "gemma2_2b_pt_l25_canonical":
        from saebase.api.sae_profiles import get_profile

        physical = get_profile(key).resolved_physical_path
        if physical:
            return physical.rstrip("/") + "/params.npz"
    if key == "gemma2_9b_pt_scope_resid":
        return p["source_file_id"].rstrip("/") + "/params.npz"
    if key == "llama31_8b_lxr_8x":
        return p["source_file_id"].rstrip("/") + "/checkpoints/final.safetensors"
    if key.startswith("r1_distill_llama8b_llamascope"):
        return p["source_file_id"].rstrip("/") + "/sae_weights.safetensors"
    if key == "gemma2_2b_it_matryoshka_l12_32k":
        return "standard/blocks.12.hook_resid_post/sae_weights.safetensors"
    if key == "qwen35_4b_matryoshka_l15_65k":
        return "qwen-3.5-4b/btk-mat-layer-15-k-100/sae_weights.safetensors"
    if key == "mistral7b_instruct_tc_l16_131k":
        return "sae_weights.safetensors"
    # SAE Lens alias resolution may have a different physical folder and
    # filename. It will be validated by the genuine CPU SAE loader instead.
    return None


def check_sources(catalog_path: str | Path = CATALOG) -> dict:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    catalog = load_coverage(catalog_path)
    records = []
    for entry in catalog.entries:
        for layer in entry.selected_layers:
            item = entry.source(layer)
            file = probe_file_path(entry, layer)
            data = {
                "sae_key": entry.sae_key,
                "layer": layer,
                "repo_id": item["repo_id"],
                "source_file": file,
                "status": "unresolved_path" if file is None else "pending",
            }
            if file:
                try:
                    metadata = get_hf_file_metadata(
                        hf_hub_url(repo_id=item["repo_id"], filename=file)
                    )
                    data.update(
                        {
                            "status": "source_file_found",
                            "size_bytes": metadata.size,
                            "etag": metadata.etag,
                            "commit_hash": metadata.commit_hash,
                        }
                    )
                except Exception as exc:
                    data.update(
                        {
                            "status": "source_check_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:350],
                        }
                    )
            records.append(data)
    return {
        "schema": "saebase.coverage.source_metadata.v1",
        "kind": "upstream_file_presence_not_weight_validation",
        "total": len(records),
        "found": sum(x["status"] == "source_file_found" for x in records),
        "unresolved": sum(x["status"] == "unresolved_path" for x in records),
        "failed": sum(x["status"] == "source_check_failed" for x in records),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": records,
    }


SCHEMA_TARGETS = "saebase.coverage.stage_d_targets.v1"

SCHEMA_STATUS = "saebase.stage_d.status.v1"

SCHEMA_ACCEPTANCE = "saebase.stage_d.pair_acceptance.v1"

SCHEMA_ACCESS = "saebase.stage_d.access_preflight.v1"

SCHEMA_FINAL = "saebase.stage_d.final_matrix.v1"

IDENTITY_MAX_ABS = 1e-2

FEATURE_DELTA_MIN = 1e-8

HEX64 = re.compile(r"^[0-9a-f]{64}$")

PRIOR_PAIR_VERIFIED: frozenset[tuple[str, str, int]] = frozenset(
    {
        ("pythia410m", "pythia410m_mlp_topk", 4),
        ("qwen3_1_7b_base", "qwen3_1_7b_base_resid_w32k", 7),
        ("r1_distill_qwen15b", "r1_distill_qwen15b_resid_l20", 20),
        ("qwen35_2b", "qwen35_2b_resid_w32k", 12),
    }
)


PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "A scientist explained the results carefully.",
    "Small birds fly across the morning sky.",
)

PROTOCOL_PAIR = "stage_d.pair.v2"

PROTOCOL_SMOKE = "stage_d.smoke_only.v1"

PROTOCOL_POPULATION = "stage_d.population.v1"

POPULATION_MIN_EVAL_GROUPS = 64

DEFAULT_POPULATION_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "coverage" / "stage_d_population.jsonl"
)

STATUS_VALUES = frozenset(
    {
        "registered",
        "sae_only_verified",
        "pair_verified",
        "benchmark_verified",
        "reliability_verified",
        "blocked_access",
        "blocked_compute",
        "blocked_loader",
        "failed_numeric",
        "failed_integrity",
        "skipped_prior",
    }
)


@dataclass(frozen=True)
class SiteKey:
    model_key: str
    sae_key: str
    layer: int
    hook_name: str

    @property
    def id(self) -> str:
        return f"{self.model_key}|{self.sae_key}|L{self.layer}|{self.hook_name}"

    def as_tuple(self) -> tuple[str, str, int]:
        return (self.model_key, self.sae_key, self.layer)


@dataclass
class SiteRecord:
    model_key: str
    sae_key: str
    layer: int
    hook_name: str
    access: str
    tier: str
    architecture: str
    size: str
    repo_id: str
    source_file_id: str
    status: str = "registered"
    prior_pair_verified: bool = False
    skip_by_default: bool = False
    execution_wave: str = "open_small"
    last_attempt_utc: str | None = None
    acceptance_path: str | None = None
    error_type: str | None = None
    error: str | None = None
    sae_state_sha256: str | None = None
    model_state_sha256: str | None = None
    checkpoint_sha: str | None = None

    @property
    def key(self) -> SiteKey:
        return SiteKey(self.model_key, self.sae_key, self.layer, self.hook_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SiteRecord":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})


def _wave_for(model_key: str, access: str) -> str:
    if model_key == "qwen35_2b":
        return "d1_calibration"
    if access in {"gated_model", "gated_large_model"}:
        return "gated"
    if model_key in {
        "qwen3_8b",
        "mistral7b_instruct",
        "r1_distill_llama8b",
        "llama31_8b",
        "qwen35_9b",
        "gemma2_9b_pt",
    }:
        return "large"
    return "open_small"


def build_site_records(catalog: CoverageCatalog) -> list[SiteRecord]:
    """Expand the Stage B/C catalog into 48 joinable site records."""
    records: list[SiteRecord] = []
    for entry in catalog.entries:
        for layer in entry.selected_layers:
            source = entry.source(layer)
            prior = (entry.model_key, entry.sae_key, layer) in PRIOR_PAIR_VERIFIED
            records.append(
                SiteRecord(
                    model_key=entry.model_key,
                    sae_key=entry.sae_key,
                    layer=layer,
                    hook_name=source["hook_name"],
                    access=entry.access,
                    tier=entry.tier,
                    architecture=entry.architecture,
                    size=entry.size,
                    repo_id=str(source["repo_id"]),
                    source_file_id=str(source["source_file_id"]),
                    status="skipped_prior" if prior else "registered",
                    prior_pair_verified=prior,
                    skip_by_default=prior,
                    execution_wave=_wave_for(entry.model_key, entry.access),
                )
            )
    if len(records) != 48:
        raise ValueError(f"expected 48 sites, got {len(records)}")
    ids = [r.key.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Stage D site ids")
    return records


DEFAULT_MODEL_ORDER: tuple[str, ...] = (
    "qwen35_2b",  # Single-site calibration
    "pythia410m",
    "qwen3_1_7b_base",
    "qwen3_1_7b",
    "qwen35_4b",
    "qwen3_8b",
    "mistral7b_instruct",
    "r1_distill_llama8b",
    "qwen35_9b",
    "gemma270m",
    "gemma1b",
    "gemma2_2b_base",
    "gemma2_2b_it",
    "gemma3_4b_pt",
    "gemma3_4b_it",
    "llama31_8b",
    "gemma2_9b_pt",
    "r1_distill_qwen15b",  # already verified; last so resume skips early
)

if __name__ == "__main__":
    raise SystemExit(main())
