"""Stage A: six public Model-SAE pair definitions and real acceptance gates.

A registry entry is not a verified checkpoint. Offline preflight never downloads
weights; the real runner hashes actual loaded model and SAE tensors and checks
capture, reconstruction, identity patching and a nontrivial SAE feature edit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
from typing import Any, Callable

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
from saebase.sae.registry import get_profile
from saebase.store import fingerprint, sha256_state_dict


SCHEMA = "saebase.stage_a.coverage.v1"


@dataclass(frozen=True)
class StageAPair:
    model_key: str
    sae_key: str
    layer: int
    model_id: str
    sae_repo: str
    family: str
    scale: str
    model_license: str
    sae_license: str
    requires_license_acceptance: bool
    d_in: int
    d_sae: int

    @property
    def site(self) -> ActivationSite:
        return ActivationSite(self.layer, f"blocks.{self.layer}.hook_resid_post")

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "hook_name": self.site.hook_name,
            "checkpoint_id": get_profile(self.sae_key).sae_id(self.layer),
        }


STAGE_A = (
    StageAPair("gemma270m", "gemma270m_scope_resid", 9,
               "google/gemma-3-270m", "google/gemma-scope-2-270m-pt",
               "gemma3", "270M", "gemma", "cc-by-4.0", True, 640, 16384),
    StageAPair("gemma1b", "gemma1b_scope_resid", 13,
               "google/gemma-3-1b-pt", "google/gemma-scope-2-1b-pt",
               "gemma3", "1B", "gemma", "cc-by-4.0", True, 1152, 16384),
    StageAPair("gemma2_2b_base", "gemma2_2b_stagea_resid_l25_16k", 25,
               "google/gemma-2-2b", "google/gemma-scope-2b-pt-res",
               "gemma2", "2B", "gemma", "cc-by-4.0", True, 2304, 16384),
    StageAPair("qwen35_2b", "qwen35_2b_resid_w32k", 12,
               "Qwen/Qwen3.5-2B-Base", "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
               "qwen3.5", "2B", "apache-2.0", "qwen", False, 2048, 32768),
    StageAPair("qwen3_8b", "qwen3_8b_resid_w64k", 18,
               "Qwen/Qwen3-8B-Base", "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
               "qwen3", "8B", "apache-2.0", "qwen", False, 4096, 65536),
    StageAPair("llama31_8b", "llama31_8b_lxr_8x", 16,
               "meta-llama/Llama-3.1-8B", "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
               "llama3.1", "8B", "llama3.1", "check upstream", True, 4096, 32768),
)

PROMPTS = (
    "The capital of France is",
    "A short explanation of neural networks is",
)


def get_pair(model_key: str) -> StageAPair:
    for item in STAGE_A:
        if item.model_key == model_key:
            return item
    raise ValueError(f"not in Stage A matrix: {model_key}; available {[p.model_key for p in STAGE_A]}")


def plan_stage_a(selected: tuple[str, ...] | None = None) -> dict:
    """Check locally registered profile contracts without running model downloads."""
    requested = tuple(p.model_key for p in STAGE_A) if selected is None else selected
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Stage A requires a nonempty selection of unique models")
    entries = []
    for name in requested:
        item = get_pair(name)
        sae = get_profile(item.sae_key)
        if sae.model_key != item.model_key or item.layer not in sae.layers:
            raise ValueError(f"{name}: SAE registry model/layer mismatch")
        if sae.hook_name(item.layer) != item.site.hook_name:
            raise ValueError(f"{name}: expected full post-layer residual SAE")
        if sae.repo_id != item.sae_repo:
            raise ValueError(f"{name}: SAE upstream repo does not match selected checkpoint")
        entries.append({**item.to_dict(), "status": "registered_not_weight_verified"})
    return {
        "schema": SCHEMA,
        "mode": "offline_preflight",
        "six_model_roster": [p.model_key for p in STAGE_A],
        "requested": list(requested),
        "pairs": entries,
        "requires_real_runs_for_verification": True,
    }


def _load_actual_model(item: StageAPair, device: str) -> ModelAdapter:
    """HF CausalLM text backbone avoids TL conversion for Qwen3.5 DeltaNet."""
    from models.hf_hooks_backend import load_hf_hooks_model
    dtype = torch.float32 if item.model_key == "gemma270m" else torch.bfloat16
    handle = load_hf_hooks_model(
        item.model_id, key=item.model_key, family=item.family,
        device=device, dtype=dtype,
    )
    return ModelAdapter(handle.model)


def _load_actual_sae(item: StageAPair, device: str):
    from saebase.sae.loading import load_sae_for_profile
    sae, metadata = load_sae_for_profile(item.sae_key, item.layer, device=device)
    return sae, metadata


def probe_pair(
    item: StageAPair, model: ModelAdapter, sae: Any,
    *, prompts: tuple[str, ...] = PROMPTS,
) -> dict[str, Any]:
    """Test a loaded pair; used with synthetic fixtures only in offline tests."""
    if not isinstance(model, ModelAdapter):
        raise TypeError("Stage A requires a ModelAdapter")
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    if int(sae.cfg.d_in) != item.d_in or int(sae.cfg.d_sae) != item.d_sae:
        raise ValueError(
            f"wrong source checkpoint dimensions: got {sae.cfg.d_in}/{sae.cfg.d_sae}; "
            f"expected {item.d_in}/{item.d_sae}"
        )
    loaded_hook = getattr(sae.cfg, "hook_name", None)
    if loaded_hook and loaded_hook not in (
        item.site.hook_name,
        f"model.layers.{item.layer}.output",
    ):
        raise ValueError(f"SAE declares incompatible hook {loaded_hook!r}")
    rows = []
    for prompt in prompts:
        snapshot = capture_features(model, sae, site=item.site, prompt=prompt)
        original = model.forward_logits(tokens=snapshot.tokens)
        model.assert_identity_patch(tokens=snapshot.tokens, site=item.site,
                                    rtol=5e-3, atol=5e-3)
        if not bool(torch.isfinite(original).all()):
            raise ValueError("nonfinite baseline logits")
        codes = snapshot.codes.float()
        active = int(torch.count_nonzero(codes).item())
        if active < 1:
            raise ValueError("SAE had no active features on reference prompt")
        winner = int(torch.argmax(codes.abs()).item())
        edited = edit_features(sae, snapshot, replacements={winner: 0.0})
        patched = model.patch_logits(
            tokens=snapshot.tokens, site=item.site, activation=edited.activation
        )
        if patched.shape != original.shape or not bool(torch.isfinite(patched).all()):
            raise ValueError("invalid edited model logits")
        difference = float(torch.linalg.vector_norm(
            (patched.float() - original.float())
        ).item())
        if difference <= 1e-8:
            raise ValueError("feature ablation caused no measurable logit change")
        h = snapshot.activation.float()
        recon = snapshot.reconstruction.float()
        mse = float((h - recon).square().mean().item())
        rows.append({
            "token_count": int(snapshot.tokens.shape[1]),
            "activation_norm": float(h.norm().item()),
            "code_l0": active,
            "reconstruction_mse": mse,
            "ablated_feature": winner,
            "edited_logit_delta_l2": difference,
        })
    return {
        "site": {"layer": item.layer, "hook_name": item.site.hook_name, "position": -1},
        "backend": model.backend,
        "d_in": item.d_in,
        "d_sae": item.d_sae,
        "n_reference_prompts": len(rows),
        "measurements": rows,
    }


def _versions() -> dict[str, str | None]:
    out = {}
    for name in ("torch", "transformers", "sae-lens", "huggingface-hub"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_real_pair(item: StageAPair, *, device: str) -> dict:
    """Only this function may label a pair verified; always loads real checkpoints."""
    model = _load_actual_model(item, device)
    sae, meta = _load_actual_sae(item, device)
    evidence = probe_pair(item, model, sae)
    raw = getattr(model.model, "_model", None)
    if raw is None:
        raise RuntimeError("Stage A real runner did not load HF weight module")
    model_sha = sha256_state_dict(raw)
    sae_sha = sha256_state_dict(sae)
    if any(len(x) != 64 for x in (model_sha, sae_sha)):
        raise RuntimeError("Stage A requires complete checkpoint hashes")
    return {
        "schema": SCHEMA,
        "status": "real_weights_verified",
        "model_key": item.model_key,
        "sae_key": item.sae_key,
        "model_id": item.model_id,
        "sae_repo": item.sae_repo,
        "sae_release": get_profile(item.sae_key).sae_release,
        "sae_id": get_profile(item.sae_key).sae_id(item.layer),
        "model_sha256": model_sha, "sae_sha256": sae_sha,
        "license": {"model": item.model_license, "sae": item.sae_license},
        "environment": _versions(),
        "device": device,
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_protocol": "exact-layer-capture, sae-encode-decode, identity-patch, ablation-logits",
        "evidence": evidence,
    }


def write_evidence(path: str | Path, record: dict) -> Path:
    from saebase.benchmarks.results import atomic_json
    path = Path(path)
    if path.exists():
        raise FileExistsError("Stage A evidence already exists; choose a new run folder")
    with_hash = {**record, "integrity_sha256": fingerprint(record)}
    atomic_json(path, with_hash)
    return path


def read_evidence(path: str | Path) -> dict:
    import json
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Stage A evidence must be an object")
    digest = record.get("integrity_sha256")
    body = {k: v for k, v in record.items() if k != "integrity_sha256"}
    if fingerprint(body) != digest:
        raise ValueError("Stage A evidence integrity mismatch")
    if record.get("status") == "real_weights_verified":
        if record.get("schema") != SCHEMA:
            raise ValueError("invalid Stage A evidence schema")
        if any(len(record.get(k, "")) != 64 for k in ("model_sha256", "sae_sha256")):
            raise ValueError("verified Stage A record missing real checkpoint digests")
        if record["evidence"]["n_reference_prompts"] < 2:
            raise ValueError("Stage A requires at least two real prompt probes")
    return record
