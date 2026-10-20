"""Paths and environment for Work Plan 05. All artifacts stay under experiments/."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "apps" / "research-lab"
POINTER = REPO / "experiments" / "research-lab" / "active_release.txt"
PACKAGES = (
    "GPU-00",
    "GPU-01",
    "GPU-02",
    "GPU-03",
    "GPU-04",
    "GPU-05",
    "GPU-06",
    "GPU-07",
    "GPU-08",
    "GPU-09",
)
INCOMPLETE_SITES = {
    "pythia410m:pythia410m_mlp_topk:L4",
    "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L7",
    "r1_distill_qwen15b:r1_distill_qwen15b_resid_l20:L20",
}
SMALL_SITE = "pythia410m:pythia410m_mlp_topk:L12"
DEEP_SITE = "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def release_root() -> Path:
    if POINTER.exists():
        path = Path(POINTER.read_text(encoding="utf-8").strip())
        if path.is_dir():
            return path
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = REPO / "experiments" / "research-lab" / f"{git_commit()[:12]}-{stamp}"
    (root / "private").mkdir(parents=True)
    (root / "tmp").mkdir()
    (root / "cache" / "hf").mkdir(parents=True)
    for name in PACKAGES:
        (root / name).mkdir()
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(str(root) + "\n", encoding="utf-8")
    return root


def free_gib(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / (1024**3)


def apply_env(root: Path) -> None:
    os.environ["TMPDIR"] = str(root / "tmp")
    os.environ["HF_HOME"] = str(root / "cache" / "hf")
    os.environ["HF_HUB_CACHE"] = str(root / "cache" / "hf" / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["TRANSFORMERS_CACHE"] = str(root / "cache" / "hf" / "transformers")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(root: Path, message: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
    print(line, flush=True)
    with (root / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def gpu_snapshot() -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, util, name = [part.strip() for part in raw.split(",", 2)]
    return {
        "time": time.time(),
        "memory_used_mib": int(memory),
        "utilization_gpu": int(util),
        "name": name,
    }


def gpu_is_idle(sample: dict[str, Any]) -> bool:
    return sample["memory_used_mib"] < 500 and sample["utilization_gpu"] < 5


def ensure_gpu_idle(root: Path, first: dict[str, Any]) -> dict[str, Any]:
    """Use the GPU only after an idle sample has stayed idle for 5 minutes."""
    sample = gpu_snapshot()
    write_json(root / "GPU-00" / "gpu_sample_latest.json", sample)
    while True:
        if not gpu_is_idle(sample):
            log(root, f"GPU busy ({sample['memory_used_mib']} MiB); waiting 5 minutes")
            time.sleep(300)
            first = gpu_snapshot()
            sample = first
            write_json(root / "GPU-00" / "gpu_sample_latest.json", sample)
            continue
        if not gpu_is_idle(first):
            first = sample
        if sample["time"] - first["time"] >= 300:
            log(root, "GPU idle for 5 minutes; proceeding")
            return sample
        wait = max(0.0, 300 - (sample["time"] - first["time"]))
        log(root, f"GPU idle; waiting {wait:.0f}s to confirm the 5-minute window")
        time.sleep(wait)
        sample = gpu_snapshot()
        write_json(root / "GPU-00" / "gpu_sample_latest.json", sample)
