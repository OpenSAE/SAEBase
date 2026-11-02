"""Stage D GPU pair acceptance ledger, scheduling, and numerical protocol.

The catalog truth source remains ``conf/coverage/stage_bc.json``. This module
adds resumable execution state and the pair-acceptance contract from the Stage D
handoff plan. Historical A/B/C evidence is joined, never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tasks.benchmark.backend.results import atomic_json
from tasks.benchmark.coverage_contract import CoverageCatalog, load_coverage
from saebase.api.sae_profiles import get_profile

SCHEMA_TARGETS = "saebase.coverage.stage_d_targets.v1"
SCHEMA_STATUS = "saebase.stage_d.status.v1"
SCHEMA_ACCEPTANCE = "saebase.stage_d.pair_acceptance.v1"
SCHEMA_ACCESS = "saebase.stage_d.access_preflight.v1"
SCHEMA_FINAL = "saebase.stage_d.final_matrix.v1"

IDENTITY_MAX_ABS = 1e-2
FEATURE_DELTA_MIN = 1e-8
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Sites already accepted on CPU with real pair + M3/M4 (Stage B) or Stage A.
PRIOR_PAIR_VERIFIED: frozenset[tuple[str, str, int]] = frozenset(
    {
        ("pythia410m", "pythia410m_mlp_topk", 4),
        ("qwen3_1_7b_base", "qwen3_1_7b_base_resid_w32k", 7),
        ("r1_distill_qwen15b", "r1_distill_qwen15b_resid_l20", 20),
        ("qwen35_2b", "qwen35_2b_resid_w32k", 12),
    }
)

# Preferred model-family order for one-at-a-time GPU sweeps.
DEFAULT_MODEL_ORDER: tuple[str, ...] = (
    "qwen35_2b",  # D1 calibration
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
    Path(__file__).resolve().parents[2] / "conf" / "coverage" / "stage_d_population.jsonl"
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


def schedule_sites(
    records: Iterable[SiteRecord],
    *,
    model: str | None = None,
    sae_key: str | None = None,
    layer: int | None = None,
    include_prior: bool = False,
    model_order: tuple[str, ...] = DEFAULT_MODEL_ORDER,
) -> list[SiteRecord]:
    """Return sites in execution order, optionally filtered.

    Prior CPU-verified sites are skipped unless ``include_prior`` is set or the
    caller names an exact ``sae_key`` + ``layer`` (D1 calibration).
    """
    explicit_site = sae_key is not None and layer is not None
    selected = []
    for rec in records:
        if model is not None and rec.model_key != model:
            continue
        if sae_key is not None and rec.sae_key != sae_key:
            continue
        if layer is not None and rec.layer != layer:
            continue
        if rec.skip_by_default and not include_prior and not explicit_site:
            continue
        selected.append(rec)

    order_index = {name: i for i, name in enumerate(model_order)}

    def sort_key(rec: SiteRecord) -> tuple:
        return (
            order_index.get(rec.model_key, len(order_index)),
            rec.sae_key,
            rec.layer,
        )

    selected.sort(key=sort_key)
    return selected


def write_stage_d_targets(path: Path, catalog: CoverageCatalog | None = None) -> dict[str, Any]:
    """Materialize the Stage D overlay from the B/C catalog."""
    if catalog is None:
        root = Path(__file__).resolve().parents[2]
        catalog = load_coverage(root / "conf" / "coverage" / "stage_bc.json")
    records = build_site_records(catalog)
    payload = {
        "schema": SCHEMA_TARGETS,
        "source_catalog_schema": "saebase.coverage.stage_bc.v1",
        "join_keys": ["model_key", "sae_key", "layer", "hook_name", "checkpoint_sha"],
        "prior_pair_verified": [
            {"model_key": m, "sae_key": s, "layer": layer}
            for m, s, layer in sorted(PRIOR_PAIR_VERIFIED)
        ],
        "model_order": list(DEFAULT_MODEL_ORDER),
        "n_sites": len(records),
        "sites": [r.to_dict() for r in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload


def load_stage_d_targets(path: Path) -> list[SiteRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_TARGETS:
        raise ValueError("invalid stage_d_targets schema")
    records = [SiteRecord.from_dict(x) for x in raw["sites"]]
    if len(records) != 48:
        raise ValueError(f"stage_d_targets must list 48 sites, got {len(records)}")
    return records


def acceptance_path(output: Path, model_key: str, sae_key: str, layer: int) -> Path:
    return output / "acceptance" / model_key / sae_key / f"L{layer}.json"


def resolve_evaluation_host(
    *,
    catalog_model_key: str,
    sae_key: str,
    model_key_override: str | None = None,
) -> tuple[str, str | None]:
    """Return (evaluation_host_model_key, model_key_override_to_pass).

    When the catalog host differs from the profile's default ``model_key`` and
    the catalog host is an allowed transfer target, automatically override so
    transfer SAEs (e.g. Gemma2 Matryoshka PT→IT) load the catalog host.
    """
    profile = get_profile(sae_key)
    if model_key_override is not None:
        if model_key_override not in profile.allowed_model_keys():
            raise ValueError(
                f"model_key_override {model_key_override!r} not in allowed keys "
                f"{sorted(profile.allowed_model_keys())}"
            )
        return model_key_override, model_key_override
    if catalog_model_key != profile.model_key and catalog_model_key in profile.allowed_model_keys():
        return catalog_model_key, catalog_model_key
    return catalog_model_key, None


def assert_host_matches_label(
    *,
    evaluation_host_model_key: str,
    loaded_model_name: str | None,
) -> None:
    """Reject IT-labeled evidence when the loaded HF repo is clearly Base."""
    if loaded_model_name is None:
        return
    name = str(loaded_model_name).lower()
    host = evaluation_host_model_key.lower()
    if host.endswith("_it") or host.endswith("-it") or "instruct" in host:
        if name.endswith("-it") or name.endswith("_it") or "instruct" in name:
            return
        # Gemma / Llama instruct hosts must not silently load their base repos.
        if name in {"google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-3-4b-pt"}:
            raise ValueError(
                f"evaluation host {evaluation_host_model_key!r} labeled instruct/IT "
                f"but loaded_model_name={loaded_model_name!r} is a base checkpoint"
            )
        if "-it" not in name and "_it" not in name and "instruct" not in name:
            # Strict for explicit IT catalog keys.
            if evaluation_host_model_key.endswith("_it"):
                raise ValueError(
                    f"evaluation host {evaluation_host_model_key!r} requires an IT "
                    f"checkpoint, got loaded_model_name={loaded_model_name!r}"
                )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_population_corpus(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen Stage D population JSONL (selection + evaluation)."""
    corpus_path = Path(path) if path is not None else DEFAULT_POPULATION_PATH
    if not corpus_path.is_file():
        raise FileNotFoundError(f"population corpus missing: {corpus_path}")
    selection: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    for line_no, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split")
        group_id = row.get("group_id")
        prompt = row.get("prompt")
        if split not in {"selection", "evaluation"}:
            raise ValueError(f"{corpus_path}:{line_no}: invalid split")
        if not isinstance(group_id, str) or not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{corpus_path}:{line_no}: invalid group/prompt")
        bucket = selection if split == "selection" else evaluation
        bucket.append({"group_id": group_id, "prompt": prompt})
    sel_groups = {r["group_id"] for r in selection}
    eval_groups = {r["group_id"] for r in evaluation}
    sel_prompts = {r["prompt"] for r in selection}
    eval_prompts = {r["prompt"] for r in evaluation}
    if not selection:
        raise ValueError("population corpus has empty selection split")
    if len(eval_groups) < POPULATION_MIN_EVAL_GROUPS:
        raise ValueError(
            f"population evaluation groups {len(eval_groups)} < {POPULATION_MIN_EVAL_GROUPS}"
        )
    if len(eval_groups) != len(evaluation):
        raise ValueError("duplicate evaluation group_id values")
    if len(sel_groups) != len(selection):
        raise ValueError("duplicate selection group_id values")
    if sel_groups & eval_groups:
        raise ValueError("selection and evaluation group_id sets overlap")
    if sel_prompts & eval_prompts:
        raise ValueError("selection and evaluation prompts overlap")
    digest = file_sha256(corpus_path)
    return {
        "path": str(corpus_path),
        "dataset_sha256": digest,
        "selection": selection,
        "evaluation": evaluation,
        "n_selection": len(selection),
        "n_evaluation_groups": len(evaluation),
        "protocol": PROTOCOL_POPULATION,
    }


def relative_under(output: Path, path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(output.resolve()))
    except Exception:
        return str(p)


def rebuild_canonical_status(
    output: Path, records: list[SiteRecord] | None = None
) -> dict[str, Any]:
    """Rewrite status.json to exactly the 48 catalog keys; archive extras."""
    if records is None:
        root = Path(__file__).resolve().parents[2]
        records = build_site_records(load_coverage(root / "conf" / "coverage" / "stage_bc.json"))
    expected = {r.key.id: r for r in records}
    if len(expected) != 48:
        raise ValueError(f"expected 48 sites, got {len(expected)}")
    old = load_status(output)
    sites: dict[str, Any] = {}
    for site_id, rec in expected.items():
        prev = dict(old.get("sites", {}).get(site_id, {}))
        acceptance = acceptance_path(output, rec.model_key, rec.sae_key, rec.layer)
        pair_ok = False
        if acceptance.exists():
            try:
                raw = read_acceptance(acceptance)
                pair_ok = raw.get("status") == "passed"
            except ValueError:
                pair_ok = False
        if not pair_ok and rec.prior_pair_verified:
            pair_ok = True
            prev.setdefault("status", "skipped_prior")
            prev.setdefault("pair_verified", True)
            prev.setdefault("prior_pair_verified", True)
        if pair_ok:
            prev["pair_verified"] = True
            if prev.get("status") in {None, "registered", "sae_only_verified"}:
                prev["status"] = (
                    "pair_verified"
                    if not rec.prior_pair_verified
                    else prev.get("status", "skipped_prior")
                )
            if rec.prior_pair_verified and prev.get("status") == "registered":
                prev["status"] = "skipped_prior"
        for key in ("acceptance_path", "m3m4_summary"):
            if key in prev:
                prev[key] = relative_under(output, prev[key])
        sites[site_id] = prev
    extras = {k: v for k, v in old.get("sites", {}).items() if k not in expected}
    if extras:
        archive = {
            "schema": "saebase.stage_d.status_archive.v1",
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "sites": extras,
        }
        atomic_json(output / "status_archive_extras.json", archive)
        append_attempt(
            output,
            {
                "event": "status_rebuild",
                "archived_extra_keys": sorted(extras),
                "n_canonical": 48,
            },
        )
    payload = {
        "schema": SCHEMA_STATUS,
        "sites": sites,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_sites": 48,
    }
    atomic_json(output / "status.json", payload)
    return payload


def write_checksums_manifest(root: Path, paths: Iterable[Path], dest: Path) -> Path:
    """Write deterministic sha256sum-compatible listing (excludes dest itself)."""
    root = root.resolve()
    dest = dest.resolve()
    rows: list[tuple[str, str]] = []
    for path in paths:
        p = path.resolve()
        if p == dest or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        rows.append((file_sha256(p), rel))
    rows.sort(key=lambda x: x[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{digest}  {rel}\n" for digest, rel in rows)
    dest.write_text(text, encoding="utf-8")
    return dest


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    encoded = json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def redact_secrets(message: str) -> str:
    text = message
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPEN_BUTTON_TOKEN", "WEB_PASSWORD"):
        token = os.environ.get(key)
        if token:
            text = text.replace(token, "[redacted]")
    return text


def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in {"GatedRepoError", "RepositoryNotFoundError"} or "401" in msg or "403" in msg:
        return "blocked_access"
    if "out of memory" in msg or "cuda error: out of memory" in msg or name == "OutOfMemoryError":
        return "blocked_compute"
    if name in {"ImportError", "ModuleNotFoundError", "UnsupportedCapability"}:
        return "blocked_loader"
    if (
        "could not load any model candidate" in msg
        or "not found. valid official model names" in msg
    ):
        return "blocked_loader"
    if name in {"ValueError", "AssertionError", "RuntimeError"} and (
        "identity" in msg or "dimension" in msg or "nonfinite" in msg or "feature" in msg
    ):
        return "failed_numeric"
    if "sha" in msg or "integrity" in msg or "manifest" in msg:
        return "failed_integrity"
    if name in {"OSError", "HTTPError", "HfHubHTTPError"}:
        return (
            "blocked_access" if any(x in msg for x in ("401", "403", "gated")) else "blocked_loader"
        )
    return "failed_numeric"


def append_attempt(output: Path, record: dict[str, Any]) -> None:
    path = output / "attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def load_status(output: Path) -> dict[str, Any]:
    path = output / "status.json"
    if not path.exists():
        return {
            "schema": SCHEMA_STATUS,
            "sites": {},
            "updated_at_utc": None,
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_STATUS:
        raise ValueError("corrupt status.json schema")
    return raw


def update_status(output: Path, site_id: str, patch: dict[str, Any]) -> None:
    status = load_status(output)
    current = dict(status["sites"].get(site_id, {}))
    current.update(patch)
    status["sites"][site_id] = current
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "status.json", status)


def read_acceptance(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_ACCEPTANCE:
        raise ValueError(f"invalid acceptance schema at {path}")
    digest = raw.get("artifact_sha256")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        raise ValueError("acceptance artifact missing SHA-256")
    expected = canonical_payload_sha256(raw)
    if digest != expected:
        raise ValueError("acceptance artifact checksum mismatch")
    return raw


def resume_fingerprint_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match for a source-aware resume skip."""
    return {
        "model_key": record.get("model_key"),
        "evaluation_host_model_key": record.get("evaluation_host_model_key")
        or record.get("model_key"),
        "sae_key": record.get("sae_key"),
        "layer": record.get("layer"),
        "hook_name": record.get("hook_name"),
        "model_state_sha256": record.get("model_state_sha256"),
        "sae_state_sha256": record.get("sae_state_sha256"),
        "model_weight_dtype": record.get("model_weight_dtype") or record.get("precision"),
        "sae_weight_dtype": record.get("sae_weight_dtype") or record.get("precision"),
        "protocol": record.get("protocol") or PROTOCOL_PAIR,
        "dataset_sha256": record.get("dataset_sha256"),
        "git_commit": record.get("git_commit"),
        "loaded_model_name": record.get("loaded_model_name"),
    }


def should_skip_resume(
    output: Path,
    *,
    model_key: str,
    sae_key: str,
    layer: int,
    resume: bool,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return existing acceptance record when resume fingerprint still matches."""
    if not resume:
        return None
    path = acceptance_path(output, model_key, sae_key, layer)
    if not path.exists():
        return None
    try:
        record = read_acceptance(path)
    except ValueError:
        return None
    if record.get("status") != "passed":
        return None
    if (
        record.get("model_key") != model_key
        or record.get("sae_key") != sae_key
        or record.get("layer") != layer
    ):
        return None
    if expected:
        got = resume_fingerprint_fields(record)
        for key, want in expected.items():
            if want is None:
                continue
            if got.get(key) != want:
                append_attempt(
                    output,
                    {
                        "event": "resume_fingerprint_mismatch",
                        "site_id": f"{model_key}|{sae_key}|L{layer}",
                        "field": key,
                        "expected": want,
                        "found": got.get(key),
                    },
                )
                return None
    return record


def verify_pair_site(
    *,
    sae_key: str,
    layer: int,
    device: str = "cuda",
    prompts: tuple[str, ...] = PROMPTS,
    model_key_override: str | None = None,
    catalog_model_key: str | None = None,
    load_fn=None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run the Stage D numerical pair protocol for one catalog site.

    ``load_fn`` is injectable for CPU unit tests. The default path loads real
    weights through ``load_model_and_sae``.
    """
    import torch

    from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
    from saebase.api.sae_loading import load_model_and_sae
    from saebase.store import sha256_state_dict

    profile = get_profile(sae_key)
    if layer not in profile.layers:
        raise ValueError(f"layer {layer} not registered for {sae_key}")
    host_key = catalog_model_key or profile.model_key
    evaluation_host, override = resolve_evaluation_host(
        catalog_model_key=host_key,
        sae_key=sae_key,
        model_key_override=model_key_override,
    )
    loader = load_fn or load_model_and_sae
    raw_model, sae, metadata = loader(
        sae_key,
        layer,
        device=device,
        model_key_override=override,
    )
    loaded_model_name = metadata.get("loaded_model_name")
    assert_host_matches_label(
        evaluation_host_model_key=evaluation_host,
        loaded_model_name=loaded_model_name,
    )
    model = raw_model if isinstance(raw_model, ModelAdapter) else ModelAdapter(raw_model)
    site = ActivationSite(layer, profile.hook_name(layer), position=-1)
    hook = metadata.get("hook_name")
    if hook != site.hook_name:
        raise ValueError(f"SAE hook {hook!r} disagrees with registry site {site.hook_name!r}")
    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise ValueError("SAE must expose encode/decode")
    d_in = int(sae.cfg.d_in)
    d_sae = int(sae.cfg.d_sae)
    if callable(getattr(model, "eval", None)):
        model.eval()

    probes = []
    peak_alloc = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for prompt in prompts:
        snap = capture_features(model, sae, site=site, prompt=prompt)
        x = snap.activation.detach().float().cpu()
        z = snap.codes.detach().float().cpu()
        recon = snap.reconstruction.detach().float().cpu()
        if x.shape != (d_in,) or z.ndim != 1 or int(z.numel()) != d_sae:
            raise ValueError("activation/SAE dimensions disagree")
        if recon.shape != x.shape:
            raise ValueError("reconstruction shape mismatch")
        if not (
            torch.isfinite(x).all() and torch.isfinite(z).all() and torch.isfinite(recon).all()
        ):
            raise ValueError("nonfinite SAE activations or reconstruction")
        baseline = model.forward_logits(tokens=snap.tokens).detach().float().cpu()
        patched = (
            model.patch_logits(
                tokens=snap.tokens,
                site=site,
                activation=snap.activation,
            )
            .detach()
            .float()
            .cpu()
        )
        if baseline.shape != patched.shape:
            raise ValueError("identity logits shape mismatch")
        err = float((baseline - patched).abs().max().item())
        if err > IDENTITY_MAX_ABS:
            raise ValueError(f"identity patch maximum logit error {err} > {IDENTITY_MAX_ABS}")
        if not torch.count_nonzero(z):
            raise ValueError("SAE has no active features on acceptance prompt")
        selected = int(torch.argmax(torch.abs(z)).item())
        edited = edit_features(sae, snap, replacements={selected: 0.0})
        edited_logits = (
            model.patch_logits(
                tokens=snap.tokens,
                site=site,
                activation=edited.activation,
            )
            .detach()
            .float()
            .cpu()
        )
        delta = float(torch.linalg.vector_norm(baseline - edited_logits).item())
        if not torch.isfinite(torch.tensor(delta)) or delta <= FEATURE_DELTA_MIN:
            raise ValueError("active SAE feature ablation did not measurably affect logits")
        probes.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "token_ids": snap.tokens.detach().cpu().tolist()[0],
                "l0": int(torch.count_nonzero(z).item()),
                "reconstruction_mse": float(torch.mean((x - recon) ** 2).item()),
                "identity_patch_max_abs_logit_error": err,
                "ablated_feature_id": selected,
                "feature_ablation_logit_l2": delta,
            }
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated()))

    underlying = model.model if isinstance(model, ModelAdapter) else raw_model
    model_sha = sha256_state_dict(underlying)
    sae_sha = sha256_state_dict(sae)
    sae_dtype = str(getattr(getattr(sae, "cfg", None), "dtype", "unknown"))
    try:
        model_dtype = str(next(underlying.parameters()).dtype)
    except Exception:
        model_dtype = "unknown"
    try:
        from saebase.api.sae_profiles import _BASE_SPECS

        host_repo = _BASE_SPECS[evaluation_host].model_name
    except Exception:
        host_repo = profile.model_name
    dataset_sha = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    payload = {
        "schema": SCHEMA_ACCEPTANCE,
        "status": "passed",
        "protocol": PROTOCOL_PAIR,
        "model_key": evaluation_host,
        "evaluation_host_model_key": evaluation_host,
        "trained_on_model_key": profile.trained_on_model_key or profile.model_key,
        "transfer_kind": profile.transfer_kind,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": site.hook_name,
        "model_weight_dtype": model_dtype,
        "sae_weight_dtype": sae_dtype,
        "precision": sae_dtype,
        "model_repo": host_repo,
        "sae_repo": profile.repo_id or profile.sae_release,
        "sae_file": metadata.get("sae_id") or profile.sae_id(layer),
        "loaded_model_name": loaded_model_name,
        "model_state_sha256": model_sha,
        "sae_state_sha256": sae_sha,
        "dataset_sha256": dataset_sha,
        "git_commit": git_commit,
        "d_in": d_in,
        "d_sae": d_sae,
        "n_prompts": len(probes),
        "identity_patch_max_abs_logit_error": max(
            p["identity_patch_max_abs_logit_error"] for p in probes
        ),
        "feature_ablation_logit_l2": [p["feature_ablation_logit_l2"] for p in probes],
        "probes": probes,
        "peak_cuda_allocated_bytes": peak_alloc,
        "device": device,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["artifact_sha256"] = canonical_payload_sha256(payload)
    return payload


def write_acceptance(output: Path, record: dict[str, Any]) -> Path:
    path = acceptance_path(
        output,
        record["model_key"],
        record["sae_key"],
        int(record["layer"]),
    )
    if "artifact_sha256" not in record:
        record = dict(record)
        record["artifact_sha256"] = canonical_payload_sha256(record)
    atomic_json(path, record)
    # Re-read validates checksum contract.
    read_acceptance(path)
    return path


def failure_record(
    *,
    sae_key: str,
    layer: int,
    device: str,
    exc: BaseException,
    model_key: str | None = None,
) -> dict[str, Any]:
    profile = get_profile(sae_key)
    status = classify_exception(exc)
    payload = {
        "schema": SCHEMA_ACCEPTANCE,
        "status": status,
        "model_key": model_key or profile.model_key,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": profile.hook_name(layer) if layer in profile.layers else None,
        "device": device,
        "error_type": type(exc).__name__,
        "error": redact_secrets(str(exc))[:1500],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_weights_validated": False,
    }
    payload["artifact_sha256"] = canonical_payload_sha256(payload)
    return payload


def _pair_verified_from_evidence(
    output: Path,
    rec: SiteRecord,
    site_status: dict[str, Any],
) -> bool:
    """Derive pair_verified only from acceptance artifacts or prior CPU proof."""
    if rec.prior_pair_verified:
        return True
    path_raw = site_status.get("acceptance_path")
    candidates: list[Path] = []
    if path_raw:
        p = Path(path_raw)
        candidates.append(p if p.is_absolute() else output / p)
    candidates.append(acceptance_path(output, rec.model_key, rec.sae_key, rec.layer))
    for path in candidates:
        if path.exists():
            try:
                raw = read_acceptance(path)
            except ValueError:
                continue
            if raw.get("status") == "passed" and raw.get("model_key") == rec.model_key:
                return True
    return False


def build_final_matrix(output: Path, records: list[SiteRecord]) -> dict[str, Any]:
    status = load_status(output)
    if set(status.get("sites", {})) - {r.key.id for r in records}:
        raise ValueError(
            "status.json contains non-canonical site keys; run rebuild_canonical_status"
        )
    if len(records) != 48:
        raise ValueError(f"final matrix requires 48 site records, got {len(records)}")
    rows = []
    counts: dict[str, int] = {}
    for rec in records:
        site_status = status["sites"].get(rec.key.id, {})
        state = site_status.get("status", rec.status)
        pair_ok = _pair_verified_from_evidence(output, rec, site_status)
        counts[state] = counts.get(state, 0) + 1
        rows.append(
            {
                "model_key": rec.model_key,
                "sae_key": rec.sae_key,
                "layer": rec.layer,
                "hook_name": rec.hook_name,
                "access": rec.access,
                "execution_wave": rec.execution_wave,
                "prior_pair_verified": rec.prior_pair_verified,
                "status": state,
                "pair_verified": pair_ok,
                "benchmark_verified": bool(site_status.get("benchmark_verified")),
                "reliability_verified": bool(site_status.get("reliability_verified")),
                "protocol": site_status.get("protocol"),
                "acceptance_path": relative_under(output, site_status.get("acceptance_path")),
                "m3m4_summary": relative_under(output, site_status.get("m3m4_summary")),
                "error_type": site_status.get("error_type"),
                "error": site_status.get("error"),
                "sae_state_sha256": site_status.get("sae_state_sha256"),
                "model_state_sha256": site_status.get("model_state_sha256"),
            }
        )
    n_pair = sum(1 for r in rows if r["pair_verified"])
    n_pop = sum(1 for r in rows if r.get("protocol") == "population" and r["reliability_verified"])
    n_smoke = sum(
        1
        for r in rows
        if r.get("protocol") in {None, "smoke_only", PROTOCOL_SMOKE} and r["reliability_verified"]
    )
    complete = n_pair == 48 and n_pop == 48
    return {
        "schema": SCHEMA_FINAL,
        "n_sites": len(rows),
        "counts": counts,
        "n_pair_verified": n_pair,
        "n_population_m3m4": n_pop,
        "n_smoke_only_m3m4": n_smoke,
        "stage_d_complete": complete,
        "rows": rows,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
