"""Inspect local prerequisites; never download weights or declare scientific PASS."""

from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_COMMIT, ROOT, canonical_hash, get_case, sha256, write_json, read_json


def preflight(
    case_id: str, *, model_snapshot: Path | None = None, sae_snapshot: Path | None = None
) -> dict:
    case = get_case(case_id)
    modules = [
        "torch",
        "numpy",
        "hydra",
        "omegaconf",
        "transformers",
        "huggingface_hub",
        "safetensors",
    ]
    if case["profile"] != "pythia14m_inference":
        modules += ["transformer_lens", "sae_lens"]
    if case["profile"] == "gemma_training":
        modules += ["datasets"]
    versions = {}
    missing = []
    distribution = {
        "hydra": "hydra-core",
        "transformer_lens": "transformer-lens",
        "sae_lens": "sae-lens",
        "huggingface_hub": "huggingface-hub",
    }
    for module in modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(module)
            versions[module] = None
        else:
            try:
                versions[module] = importlib.metadata.version(distribution.get(module, module))
            except importlib.metadata.PackageNotFoundError:
                versions[module] = "importable_version_unknown"
    gpu = {"checked": False, "cuda_available": None}
    if "torch" not in missing:
        import torch

        gpu = {
            "checked": True,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if gpu["cuda_available"]:
            gpu["memory_bytes"] = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            ]
    blockers = ["missing_dependency:" + m for m in missing]
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        blockers.append("unsupported_python_requires_3_11_or_3_12")

    def snapshot_present(path: Path | None) -> bool:
        return bool(path and path.is_dir() and any(p.is_file() for p in path.rglob("*")))

    if not snapshot_present(model_snapshot):
        blockers.append("model_snapshot_missing_or_empty")
    if case["profile"] != "pythia14m_inference" and not snapshot_present(sae_snapshot):
        # Training uses a newly trained SAE; source SAE is not required there.
        if case["profile"] != "gemma_training" and case_id != "R09":
            blockers.append("sae_snapshot_not_supplied")
    if case["profile"] == "gemma_training" and not gpu["cuda_available"]:
        blockers.append("training_profile_requires_single_cuda_device")
    return {
        "schema": "saebase.wp21.preflight.v1",
        "case_id": case_id,
        "base_commit": BASE_COMMIT,
        "status": "ENV_BLOCKED" if blockers else "PREREQUISITES_PRESENT_UNVERIFIED",
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": versions,
        "gpu": gpu,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "blockers": blockers,
        "downloaded_weights": False,
        "executed_model": False,
        "model_snapshot": str(model_snapshot.resolve())
        if model_snapshot and model_snapshot.exists()
        else None,
        "sae_snapshot": str(sae_snapshot.resolve())
        if sae_snapshot and sae_snapshot.exists()
        else None,
        "note": "Paths/dependencies alone do not prove the correct checkpoint loaded. Freeze assets and run the case's actual model probe.",
    }


def preflight_all(
    *,
    model_snapshot: Path | None = None,
    sae_snapshot: Path | None = None,
    pythia_snapshot: Path | None = None,
) -> dict:
    """Pre-dispatch inventory covering every frozen case and both model types.

    This deliberately never downloads or infers successful forward passes.
    R09 is model-only, R17 trains its own SAE, and R08 uses a separate frozen
    Pythia checkpoint. Case-level preflight alone can miss those distinctions.
    """
    checks = []
    for n in range(1, 21):
        cid = f"R{n:02d}"
        case = get_case(cid)
        is_pythia = case["profile"] == "pythia14m_inference"
        model = pythia_snapshot if is_pythia else model_snapshot
        need_sae = not (is_pythia or case["profile"] == "gemma_training" or cid == "R09")
        result = preflight(
            cid, model_snapshot=model, sae_snapshot=sae_snapshot if need_sae else None
        )
        checks.append(
            {
                "case_id": cid,
                "profile": case["profile"],
                "requirements": {
                    "model": "pythia-14m" if is_pythia else "gemma-2-2b",
                    "sae": "gemma-scope-layer25" if need_sae else None,
                    "single_cuda_device": case["profile"] == "gemma_training",
                },
                "status": result["status"],
                "blockers": result["blockers"],
                "model_snapshot": result["model_snapshot"],
                "sae_snapshot": result["sae_snapshot"],
            }
        )
    blocked = [r["case_id"] for r in checks if r["status"] == "ENV_BLOCKED"]
    return {
        "schema": "saebase.wp21.preflight_all.v1",
        "status": "ENV_BLOCKED" if blocked else "PREREQUISITES_PRESENT_UNVERIFIED",
        "cases": checks,
        "counts": {"ready_for_probe": 20 - len(blocked), "env_blocked": len(blocked)},
        "blocked_cases": blocked,
        "downloaded_weights": False,
        "executed_model": False,
        "note": "Per-case prerequisites only; frozen revisions, loaded devices and actual model execution still require immutable asset locks and real probes.",
    }


def freeze_assets(items: list[tuple[str, Path]], output: Path) -> dict:
    """Hash caller-selected local snapshots once, retaining private paths locally."""
    files = []
    roles = [role for role, _ in items]
    if len(roles) != len(set(roles)) or not set(roles) <= {"model", "sae"}:
        raise ValueError("duplicate or unknown asset role")
    for role, directory in items:
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("asset snapshot must be a directory")
        for p in sorted(directory.rglob("*")):
            if p.is_file():
                # HF snapshots legitimately link to sibling blob storage.
                resolved = p.resolve(strict=True)
                if not resolved.is_file():
                    raise ValueError("asset is not a regular file")
                files.append(
                    {
                        "role": role,
                        "snapshot_relative_path": p.relative_to(directory).as_posix(),
                        "local_path": str(resolved),
                        "size_bytes": resolved.stat().st_size,
                        "sha256": sha256(resolved),
                    }
                )
    if not files:
        raise ValueError("empty asset snapshot")
    lock = {
        "schema": "saebase.wp21.assets.v1",
        "status": "FILES_HASHED_NOT_RUNTIME_VERIFIED",
        "files": files,
        "sha256": canonical_hash(files),
    }
    write_json(output, lock, exclusive=True)
    return {
        "status": lock["status"],
        "file_count": len(files),
        "sha256": lock["sha256"],
        "output": str(output),
    }


def verify_assets(lock_path: Path) -> dict:
    """Re-hash trusted cache files before independent replay.

    This validates bytes on disk but cannot establish that the model loader
    actually consumed them. A real probe and runtime trace remain necessary.
    """
    lock = read_json(lock_path)
    if lock.get("schema") != "saebase.wp21.assets.v1" or not isinstance(lock.get("files"), list):
        raise ValueError("invalid asset lock")
    files = lock["files"]
    if not files or canonical_hash(files) != lock.get("sha256"):
        raise ValueError("asset lock digest mismatch")
    seen = set()
    for entry in files:
        key = (entry["role"], entry["snapshot_relative_path"])
        if key in seen:
            raise ValueError("duplicate asset lock entry")
        seen.add(key)
        path = Path(entry["local_path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["size_bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError("asset bytes changed or missing: " + repr(key))
    return {
        "status": "ASSETS_UNCHANGED_NOT_RUNTIME_VERIFIED",
        "file_count": len(files),
        "sha256": lock["sha256"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="R01")
    p.add_argument(
        "--all", action="store_true", help="Audit all R01-R20 prerequisites before dispatch"
    )
    p.add_argument("--pythia-snapshot", type=Path, help="Frozen Pythia-14m model snapshot for R08")
    p.add_argument("--model-snapshot", type=Path)
    p.add_argument("--sae-snapshot", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--freeze-assets", type=Path)
    p.add_argument("--verify-assets", type=Path)
    a = p.parse_args()
    if a.freeze_assets and a.verify_assets:
        p.error("freeze and verify are mutually exclusive")
    if a.all and (a.freeze_assets or a.verify_assets):
        p.error("--all is a prerequisite inventory, not an asset freeze/verify operation")
    if a.all:
        result = preflight_all(
            model_snapshot=a.model_snapshot,
            sae_snapshot=a.sae_snapshot,
            pythia_snapshot=a.pythia_snapshot,
        )
        if a.output:
            write_json(a.output, result, exclusive=True)
    elif a.verify_assets:
        result = verify_assets(a.verify_assets)
    elif a.freeze_assets:
        items = [(k, v) for k, v in [("model", a.model_snapshot), ("sae", a.sae_snapshot)] if v]
        result = freeze_assets(items, a.freeze_assets)
    else:
        result = preflight(a.case, model_snapshot=a.model_snapshot, sae_snapshot=a.sae_snapshot)
        if a.output:
            write_json(a.output, result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 3 if result.get("status") == "ENV_BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
