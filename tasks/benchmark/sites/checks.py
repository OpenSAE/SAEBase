"""Purpose: Verify a selected real model SAE site, record numerical acceptance and organize its bounded execution.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from saebase.api.sae_profiles import get_profile
from tasks.benchmark.sites.catalog import (
    DEFAULT_POPULATION_PATH,
    FEATURE_DELTA_MIN,
    IDENTITY_MAX_ABS,
    POPULATION_MIN_EVAL_GROUPS,
    PROMPTS,
    PROTOCOL_PAIR,
    PROTOCOL_POPULATION,
    SCHEMA_ACCEPTANCE,
    load_coverage,
)
from tasks.benchmark.sites.records import (
    append_attempt,
    canonical_payload_sha256,
    failure_record,
    file_sha256,
    relative_under,
    should_skip_resume,
    update_status,
    write_acceptance,
)


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


ROOT = Path(__file__).resolve().parents[3]

CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_one(
    *,
    sae_key: str,
    layer: int,
    device: str,
    output: Path,
    resume: bool,
    model_key_override: str | None = None,
) -> dict:
    catalog = load_coverage(CATALOG)
    entry = catalog.find(sae_key)
    if layer not in entry.selected_layers:
        raise ValueError(f"layer {layer} is not a Stage B/C selected layer for {sae_key}")
    profile = get_profile(sae_key)
    evaluation_host, auto_override = resolve_evaluation_host(
        catalog_model_key=entry.model_key,
        sae_key=sae_key,
        model_key_override=model_key_override,
    )
    site_id = f"{evaluation_host}|{sae_key}|L{layer}|{profile.hook_name(layer)}"
    started = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit()
    # Fingerprint gates science-invalidating drift. Legacy pair artifacts without
    # git_commit/dataset fields still resume when host/hook/protocol match;
    # Mistral LN and Gemma IT host fixes are re-run with --no-resume.
    expected = {
        "model_key": evaluation_host,
        "evaluation_host_model_key": evaluation_host,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": profile.hook_name(layer),
        "protocol": PROTOCOL_PAIR,
    }
    skipped = should_skip_resume(
        output,
        model_key=evaluation_host,
        sae_key=sae_key,
        layer=layer,
        resume=resume,
        expected=expected,
    )
    if skipped is not None:
        update_status(
            output,
            site_id,
            {
                "status": "pair_verified",
                "pair_verified": True,
                "acceptance_path": relative_under(
                    output,
                    output / "acceptance" / evaluation_host / sae_key / f"L{layer}.json",
                ),
                "resumed": True,
                "protocol": PROTOCOL_PAIR,
                "sae_state_sha256": skipped.get("sae_state_sha256"),
                "model_state_sha256": skipped.get("model_state_sha256"),
                "evaluation_host_model_key": evaluation_host,
                "loaded_model_name": skipped.get("loaded_model_name"),
            },
        )
        append_attempt(
            output,
            {
                "site_id": site_id,
                "event": "resume_skip",
                "started_at_utc": started,
                "status": "pair_verified",
            },
        )
        return skipped

    append_attempt(
        output,
        {
            "site_id": site_id,
            "event": "start",
            "device": device,
            "started_at_utc": started,
            "evaluation_host_model_key": evaluation_host,
            "model_key_override": auto_override,
        },
    )
    try:
        record = verify_pair_site(
            sae_key=sae_key,
            layer=layer,
            device=device,
            prompts=PROMPTS,
            model_key_override=auto_override,
            catalog_model_key=entry.model_key,
            git_commit=git_commit,
        )
        record["git_commit"] = git_commit
        record["python"] = platform.python_version()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        if torch.cuda.is_available() and device.startswith("cuda"):
            record["gpu_name"] = torch.cuda.get_device_name(0)
        from tasks.benchmark.sites.records import canonical_payload_sha256

        record.pop("artifact_sha256", None)
        record["artifact_sha256"] = canonical_payload_sha256(record)
        path = write_acceptance(output, record)
        update_status(
            output,
            site_id,
            {
                "status": "pair_verified",
                "pair_verified": True,
                "acceptance_path": relative_under(output, path),
                "protocol": PROTOCOL_PAIR,
                "sae_state_sha256": record["sae_state_sha256"],
                "model_state_sha256": record["model_state_sha256"],
                "evaluation_host_model_key": evaluation_host,
                "trained_on_model_key": record.get("trained_on_model_key"),
                "transfer_kind": record.get("transfer_kind"),
                "loaded_model_name": record.get("loaded_model_name"),
                "error_type": None,
                "error": None,
            },
        )
        append_attempt(
            output,
            {
                "site_id": site_id,
                "event": "end",
                "status": "passed",
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "peak_cuda_allocated_bytes": record.get("peak_cuda_allocated_bytes"),
                "supersedes_prior": True,
            },
        )
        return record
    except Exception as exc:
        record = failure_record(
            sae_key=sae_key,
            layer=layer,
            device=device,
            exc=exc,
            model_key=evaluation_host,
        )
        record["git_commit"] = git_commit
        from tasks.benchmark.sites.records import canonical_payload_sha256

        record.pop("artifact_sha256", None)
        record["artifact_sha256"] = canonical_payload_sha256(record)
        path = write_acceptance(output, record)
        update_status(
            output,
            site_id,
            {
                "status": record["status"],
                "pair_verified": False,
                "acceptance_path": relative_under(output, path),
                "error_type": record["error_type"],
                "error": record["error"],
            },
        )
        append_attempt(
            output,
            {
                "site_id": site_id,
                "event": "end",
                "status": record["status"],
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": record["error_type"],
            },
        )
        return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-key", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/stage_d_runs"))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--model-key-override", default=None)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    record = run_one(
        sae_key=args.sae_key,
        layer=args.layer,
        device=args.device,
        output=args.output,
        resume=args.resume,
        model_key_override=args.model_key_override,
    )
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
