"""Prepare real dependencies and exact Hub assets, with durable attempt receipts.

Preparation readiness never means that model loading or a scientific check passed.
Precheck runs this function in a worker with a separate preparation timeout.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


def _redact(value: str) -> str:
    value = re.sub(r"hf_[A-Za-z0-9]{10,}", "[REDACTED_HF_TOKEN]", value)
    return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", value)


def _import_check(modules: tuple[str, ...]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib,sys; [importlib.import_module(n) for n in sys.argv[1:]]",
            *modules,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _installation_target(root: Path) -> list[str]:
    if (root / "pyproject.toml").is_file():
        return ["-e", ".[analysis]"]
    # Installed wheels retain their dependency metadata, without a source checkout.
    version = importlib.metadata.version("opensae-saebase")
    return [f"opensae-saebase[analysis]=={version}"]


def _resolve_asset(item: dict) -> dict:
    result = dict(item)
    if item.get("kind") == "sae" and item.get("loader") == "sae_lens":
        from sae_lens.loading.pretrained_saes_directory import get_repo_id_and_folder_name

        repo, folder = get_repo_id_and_folder_name(item["release"], item["sae_id"])
        result["repo_id"] = repo
        result["folder"] = folder
    if not result.get("repo_id") or "/" not in result["repo_id"]:
        raise ValueError(f"An exact Hub repository is required for {item.get('model_key', item)}")
    return result


def _files(item: dict, names: list[str]) -> list[str]:
    if item.get("kind") == "model":
        flat = [name for name in names if "/" not in name]
        weights = [
            name for name in flat if name.startswith("model") and name.endswith(".safetensors")
        ]
        if not weights:
            weights = [
                name for name in flat if name.startswith("pytorch_model") and name.endswith(".bin")
            ]
        if not weights or "config.json" not in flat:
            raise ValueError("Model repository has no supported complete weight/config files")
        metadata = [
            name for name in flat if name.endswith((".json", ".model", ".tiktoken", ".txt", ".py"))
        ]
        return sorted(set(weights + metadata))
    if item.get("folder"):
        folder = item["folder"].strip("./")
        prefix = folder + "/" if folder else ""
        matches = (
            [folder] if folder in names else [name for name in names if name.startswith(prefix)]
        )
    else:
        matches = names
    if not matches:
        raise ValueError("Selected SAE folder has no files")
    return sorted(matches)


def _cache_complete(path: Path, item: dict) -> bool:
    names = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]
    try:
        selected = _files(item, names)
    except ValueError:
        return False
    if item.get("kind") == "model":
        if not any(
            "tokenizer" in name or name.endswith(("vocab.json", ".model")) for name in selected
        ):
            return False
        for name in selected:
            if name.endswith(".index.json"):
                manifest = json.loads((path / name).read_text())
                if any(
                    not (path / shard).is_file()
                    for shard in manifest.get("weight_map", {}).values()
                ):
                    return False
    elif item.get("kind") == "sae":
        if not any(name.endswith((".npz", ".safetensors", ".pt", ".pth")) for name in selected):
            return False
    return all((path / name).is_file() for name in selected)


def prepare_environment(
    *,
    required_modules: tuple[str, ...] = (
        "transformer_lens",
        "sae_lens",
        "huggingface_hub",
        "neuronpedia",
    ),
    assets: list[dict[str, Any]] | None = None,
    allow_network: bool = True,
    allow_install: bool = True,
    allow_download: bool = True,
    max_download_bytes: int = 16 * 1024**3,
    install_timeout_seconds: int = 1800,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Attempt missing preparation by default; explicit offline settings remain effective."""
    attempted: list[str] = []
    events: list[dict] = []
    prepared: list[dict] = []
    logs = Path(log_dir) if log_dir is not None else None
    if logs is not None:
        logs.mkdir(parents=True, exist_ok=True)
    if type(max_download_bytes) is not int or max_download_bytes < 0:
        raise ValueError("max_download_bytes must be a nonnegative integer")

    def event(data):
        events.append(data)
        if logs is not None:
            from saebase.runtime.experiment_results import write_json

            write_json(logs / "events.json", events)

    def finish(status, code=None, reason=None):
        return {
            "status": status,
            "stage": "setup",
            "code": code,
            "reason": reason,
            "attempted": attempted,
            "events": events,
            "prepared_assets": prepared,
            "required_modules": list(required_modules),
            "repair": None
            if status == "READY"
            else "Inspect the preparation receipt and logs; retry the same selected resources after correcting the reported failure.",
            "log_dir": str(logs) if logs else None,
            "scientific_test": "NOT_RUN",
            "gpu_test": "NOT_RUN",
        }

    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        if not (allow_network and allow_install):
            return finish(
                "BLOCKED",
                "BLOCKED_DEPENDENCY",
                f"Missing modules {missing}; installation explicitly disabled",
            )
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="saebase-install-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            pins = []
            # Installing missing backends must not replace already-loaded numeric or model packages.
            for package in ("torch", "numpy", "transformers", "sae-lens", "huggingface-hub"):
                try:
                    pins.append(f"{package}=={importlib.metadata.version(package)}")
                except importlib.metadata.PackageNotFoundError:
                    pass
            constraints.write_text("\n".join(pins) + "\n")
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--constraint",
                str(constraints),
                *_installation_target(root),
            ]
            attempted.append(
                "install missing analysis dependencies with installed model/numeric versions constrained"
            )
            event(
                {
                    "stage": "dependencies",
                    "status": "attempting",
                    "missing": missing,
                    "constraints": pins,
                }
            )
            try:
                proc = subprocess.run(
                    command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=install_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                if logs is not None:
                    for label, content in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="replace")
                        (logs / f"pip.{label}.log").write_text(_redact(content or ""))
                event({"stage": "dependencies", "status": "timeout"})
                return finish("BLOCKED", "BLOCKED_DEPENDENCY", "Dependency installation timed out")
            if logs is not None:
                (logs / "pip.stdout.log").write_text(_redact(proc.stdout))
                (logs / "pip.stderr.log").write_text(_redact(proc.stderr))
            event(
                {
                    "stage": "dependencies",
                    "status": "installed" if proc.returncode == 0 else "failed",
                    "exit_code": proc.returncode,
                }
            )
            if proc.returncode:
                return finish(
                    "BLOCKED",
                    "BLOCKED_DEPENDENCY",
                    _redact(proc.stderr[-1500:] or proc.stdout[-1500:]),
                )
        importlib.invalidate_caches()
    if required_modules:
        try:
            checked = _import_check(required_modules)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return finish("BLOCKED", "BLOCKED_DEPENDENCY", str(exc))
        if logs is not None:
            (logs / "imports.stdout.log").write_text(_redact(checked.stdout))
            (logs / "imports.stderr.log").write_text(_redact(checked.stderr))
        event(
            {
                "stage": "dependency_imports",
                "status": "ready" if checked.returncode == 0 else "failed",
                "exit_code": checked.returncode,
            }
        )
        if checked.returncode:
            return finish("BLOCKED", "BLOCKED_DEPENDENCY", _redact(checked.stderr[-1500:]))
    if assets:
        from huggingface_hub import HfApi, snapshot_download

        remaining = max_download_bytes
        for requested in assets:
            cached = None
            try:
                item = _resolve_asset(requested)
            except Exception as exc:
                return finish(
                    "BLOCKED", "BLOCKED_CONFIGURATION", f"Cannot resolve exact resource: {exc}"
                )
            repo, revision = item["repo_id"], item.get("revision")
            try:
                cached = Path(
                    snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
                )
                if _cache_complete(cached, item):
                    prepared.append(
                        {
                            **item,
                            "cache_path": str(cached),
                            "source": "cache",
                            "resolved_revision": cached.name,
                        }
                    )
                    event(
                        {"stage": "asset", "repo_id": repo, "status": "cached", "path": str(cached)}
                    )
                    continue
            except (OSError, ValueError):
                pass
            if not (allow_network and allow_download):
                return finish(
                    "BLOCKED",
                    "BLOCKED_ASSET",
                    f"Resource {repo} is incomplete and download explicitly disabled",
                )
            attempted.append(f"download {repo}@{revision or 'resolved immutable revision'}")
            event(
                {
                    "stage": "asset",
                    "repo_id": repo,
                    "status": "attempting",
                    "requested_revision": revision,
                }
            )
            try:
                info = HfApi().model_info(repo, revision=revision, files_metadata=True)
                if not info.sha:
                    raise ValueError("Hub did not resolve an immutable revision")
                names = _files(item, [f.rfilename for f in info.siblings or []])
                sizes = {f.rfilename: f.size for f in info.siblings or []}
                if any(type(sizes[name]) is not int or sizes[name] < 0 for name in names):
                    raise ValueError("Selected resource sizes are not verifiable")
                # A SHA-pinned download need not create a mutable refs/main entry.
                # Reuse its exact snapshot even when the initial branch lookup missed it.
                if cached is None or cached.name != info.sha:
                    try:
                        cached = Path(
                            snapshot_download(
                                repo_id=repo, revision=info.sha, local_files_only=True
                            )
                        )
                    except (OSError, ValueError):
                        cached = None
                # Charge only missing selected files, not every checkpoint in the repository.
                needed = sum(
                    sizes[name]
                    for name in names
                    if not (
                        cached is not None
                        and cached.name == info.sha
                        and (cached / name).is_file()
                        and (cached / name).stat().st_size == sizes[name]
                    )
                )
                from huggingface_hub.constants import HF_HUB_CACHE

                Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
                if needed > remaining or needed > shutil.disk_usage(HF_HUB_CACHE).free:
                    event(
                        {
                            "stage": "asset",
                            "repo_id": repo,
                            "status": "budget_exceeded",
                            "required_bytes": needed,
                            "remaining_bytes": remaining,
                        }
                    )
                    return finish(
                        "BLOCKED",
                        "BLOCKED_RESOURCE",
                        f"Selected files need {needed} bytes; remaining download budget {remaining}",
                    )
                cache = Path(
                    snapshot_download(repo_id=repo, revision=info.sha, allow_patterns=names)
                )
                if any(
                    not (cache / name).is_file() or (cache / name).stat().st_size != sizes[name]
                    for name in names
                ):
                    raise ValueError("Downloaded snapshot is incomplete or file sizes disagree")
                if not _cache_complete(cache, item):
                    raise ValueError("Required model/tokenizer/SAE files are incomplete")
                remaining -= needed
                entry = {
                    **item,
                    "resolved_revision": info.sha,
                    "cache_path": str(cache),
                    "source": "download",
                    "selected_files": names,
                    "download_bytes": needed,
                }
                prepared.append(entry)
                event(
                    {
                        "stage": "asset",
                        "repo_id": repo,
                        "status": "ready",
                        "resolved_revision": info.sha,
                        "download_bytes": needed,
                    }
                )
            except Exception as exc:
                event(
                    {
                        "stage": "asset",
                        "repo_id": repo,
                        "status": "failed",
                        "error": _redact(str(exc)),
                    }
                )
                return finish(
                    "BLOCKED",
                    "BLOCKED_ASSET",
                    f"{repo}: {type(exc).__name__}: {_redact(str(exc))[:1500]}",
                )
    return finish("READY")
