"""P5 bounded native SAELens and Sparsify execution, checkpoint registration.

Never substitutes a home-built trainer for SAELens/Sparsify. Local snapshot
paths and permission must be supplied to keep dataset/model identity stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Literal

SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TrainPlan:
    backend: Literal["saelens", "sparsify"]
    model_path: str
    dataset_path: str
    model_revision: str
    dataset_revision: str
    hookpoint: str
    d_in: int
    d_sae: int
    k: int
    architecture: Literal["topk", "standard", "jumprelu", "gated", "transcoder"]
    max_examples: int
    steps: int
    batch_size: int
    ctx_len: int
    run_name: str
    device: str = "cuda"
    resume_checkpoint: str | None = None
    distributed_gpus: int = 1

    def validate(self, *, allow_large: bool = False) -> None:
        if self.backend not in {"saelens", "sparsify"}:
            raise ValueError("Unknown native trainer")
        if self.architecture == "transcoder" and self.backend != "sparsify":
            raise ValueError("This adapter supports transcoder only in native Sparsify")
        if self.backend == "sparsify" and self.architecture not in {"topk", "transcoder"}:
            raise ValueError("Sparsify backend accepts its native TopK/--transcode route")
        if any(not SHA.fullmatch(x) for x in (self.model_revision, self.dataset_revision)):
            raise ValueError("Exact pinned model and dataset commits required")
        if not (Path(self.model_path).is_dir() and Path(self.dataset_path).is_dir()):
            raise FileNotFoundError("Native training needs existing immutable local snapshots")
        if not self.hookpoint or any(x.isspace() for x in self.hookpoint):
            raise ValueError("Explicit concrete hookpoint required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", self.run_name):
            raise ValueError("Safe 3-64 character experiment name required")
        if not (
            0 < self.k <= self.d_sae and 0 < self.d_in <= 16384 and 0 < self.d_sae <= 1_048_576
        ):
            raise ValueError("Invalid native model/SAE width/sparsity")
        if not (1 <= self.batch_size <= 256 and 16 <= self.ctx_len <= 4096):
            raise ValueError("Invalid batch/context budget")
        if not (
            1 <= self.steps <= (200000 if allow_large else 200)
            and 1 <= self.max_examples <= (100_000_000 if allow_large else 10000)
        ):
            raise PermissionError("Large C2/C3 training needs separate authorization")
        if self.backend == "sparsify" and self.max_examples > self.steps * self.batch_size:
            raise ValueError(
                "Sparsify trains on dataset examples, not a step limit; max_examples "
                "must fit within the predeclared steps × batch_size budget"
            )
        if self.distributed_gpus < 1 or self.distributed_gpus > 16:
            raise ValueError("GPU world size outside supported bounds")
        if self.distributed_gpus > 1 and not allow_large:
            raise PermissionError("Multi-GPU C3 training requires explicit approval")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("Only local device execution is supported")
        if self.resume_checkpoint is not None and not Path(self.resume_checkpoint).is_dir():
            raise FileNotFoundError("Cannot resume without native checkpoint directory")


def _saelens_train(plan: TrainPlan, checkpoint_dir: Path) -> None:
    """Original SAELens v6 nested config; its trainer computes real activations."""
    from sae_lens import (  # type: ignore[import-not-found]
        LanguageModelSAERunnerConfig,
        LanguageModelSAETrainingRunner,
        LoggingConfig,
        StandardTrainingSAEConfig,
        TopKTrainingSAEConfig,
        JumpReLUTrainingSAEConfig,
        GatedTrainingSAEConfig,
    )

    implementations = {
        "standard": StandardTrainingSAEConfig,
        "topk": TopKTrainingSAEConfig,
        "jumprelu": JumpReLUTrainingSAEConfig,
        "gated": GatedTrainingSAEConfig,
    }
    config_class = implementations[plan.architecture]
    options = {"d_in": plan.d_in, "d_sae": plan.d_sae}
    if plan.architecture == "topk":
        options["k"] = plan.k
    sae = config_class(**options)
    runner = LanguageModelSAERunnerConfig(
        model_name=plan.model_path,
        hook_name=plan.hookpoint,
        dataset_path=plan.dataset_path,
        dataset_trust_remote_code=False,
        streaming=False,
        sae=sae,
        train_batch_size_tokens=plan.batch_size,
        training_tokens=plan.steps * plan.batch_size,
        context_size=plan.ctx_len,
        seed=0,
        device=plan.device,
        dtype="float32",
        n_checkpoints=1,
        checkpoint_path=str(checkpoint_dir),
        save_final_checkpoint=True,
        output_path=str(checkpoint_dir / "inference"),
        resume_from_checkpoint=plan.resume_checkpoint,
        logger=LoggingConfig(log_to_wandb=False),
    )
    LanguageModelSAETrainingRunner(runner).run()


def _sparsify_command(plan: TrainPlan) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sparsify",
        plan.model_path,
        plan.dataset_path,
        "--run_name",
        plan.run_name,
        "--hookpoints",
        plan.hookpoint,
        "--ctx_len",
        str(plan.ctx_len),
        "--max_examples",
        str(plan.max_examples),
        "--batch_size",
        str(plan.batch_size),
        "--k",
        str(plan.k),
        "--num_latents",
        str(plan.d_sae),
        "--revision",
        plan.model_revision,
    ]
    if plan.architecture == "transcoder":
        cmd.append("--transcode")
    if plan.resume_checkpoint is not None:
        # Native Sparsify restores optimizer, scheduler and trainer state with
        # --resume from checkpoints/{run_name}. --finetune only loads weights.
        cmd.append("--resume")
    if plan.distributed_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(plan.distributed_gpus),
            "-m",
            "sparsify",
            *cmd[3:],
        ]
    return cmd


def run_training(
    plan: TrainPlan,
    *,
    output: Path,
    approved: bool = False,
    allow_large: bool = False,
    timeout_seconds: int = 1800,
) -> dict:
    """Real native execution only on explicit approval and actual local snapshots."""
    plan.validate(allow_large=allow_large)
    if not approved:
        raise PermissionError("Training must be explicitly authorized")
    if output.exists():
        raise FileExistsError("Training output and historical state are immutable")
    if plan.backend == "sparsify" and plan.resume_checkpoint is not None:
        raise ValueError(
            "Use resume_sparsify_training on the existing native run directory; "
            "a new output cannot resume its old optimizer/data state"
        )
    if plan.backend == "sparsify" and plan.device != "cuda":
        raise RuntimeError("Native Sparsify training requires an actual CUDA device")
    if not 30 <= timeout_seconds <= (86400 if allow_large else 3600):
        raise ValueError("Training timeout outside approved tier")
    import torch

    if plan.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No real CUDA GPU available")
    if plan.distributed_gpus > 1 and torch.cuda.device_count() < plan.distributed_gpus:
        raise RuntimeError("Real 2+ GPU allocation required; cannot simulate distributed")
    output.mkdir(parents=True, mode=0o700)
    spec_path = output / "plan.json"
    spec_path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n")
    start = time.monotonic()
    if plan.backend == "saelens":
        _saelens_train(plan, output / "checkpoint")
        return {
            "status": "native_training_returned",
            "backend": plan.backend,
            "elapsed_seconds": time.monotonic() - start,
            "checkpoints_need_registration": True,
        }
    command = _sparsify_command(plan)
    # Source returns paths and progress according to the installed native CLI;
    # stdout/logs cannot certify which specific weights were saved.
    try:
        result = subprocess.run(
            command,
            cwd=output,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "WANDB_MODE": "disabled"},
        )
    except subprocess.TimeoutExpired:
        (output / "status.json").write_text('{"status":"timeout"}\n')
        raise
    (output / "native.stdout.log").write_text(result.stdout)
    (output / "native.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Native Sparsify exited {result.returncode}; inspect private logs")
    return {
        "status": "native_training_returned",
        "backend": plan.backend,
        "elapsed_seconds": time.monotonic() - start,
        "checkpoints_need_registration": True,
        "distributed_gpus_requested": plan.distributed_gpus,
    }



def resume_sparsify_training(
    plan: TrainPlan, *, output: Path, approved: bool = False,
    allow_large: bool = False, timeout_seconds: int = 1800,
) -> dict:
    """Resume real optimizer/trainer state only in its original run directory.

    Sparsify's upstream CLI resolves checkpoints/{run_name} relative to the
    current directory. The owner must approve resuming that *same* run; a
    fine-tuned new model is not equivalent to a resumed training trajectory.
    Logs are never overwritten, and failed attempts remain inspectable.
    """
    if plan.backend != "sparsify" or plan.resume_checkpoint is None:
        raise ValueError("Only genuine native Sparsify resume plans are supported")
    plan.validate(allow_large=allow_large)
    if not approved:
        raise PermissionError("Explicit checkpoint-resume permission required")
    if not 30 <= timeout_seconds <= (86400 if allow_large else 3600):
        raise ValueError("Resume timeout outside approved budget")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Sparsify resume requires actual CUDA hardware")
    if plan.distributed_gpus > torch.cuda.device_count():
        raise RuntimeError("Cannot simulate missing distributed CUDA ranks")
    output = output.resolve()
    expected = (output / "checkpoints" / plan.run_name).resolve()
    if Path(plan.resume_checkpoint).resolve() != expected:
        raise ValueError("Resume checkpoint must be native checkpoints/{run_name}")
    if not (expected / "state.pt").is_file():
        raise FileNotFoundError("Native optimizer/trainer state.pt is missing")
    original = output / "plan.json"
    if not original.is_file():
        raise FileNotFoundError("Original frozen native training plan is missing")
    prior = json.loads(original.read_text(encoding="utf-8"))
    current = asdict(plan)
    for record in (prior, current):
        record.pop("resume_checkpoint", None)
    if prior != current:
        raise ValueError("Full native resume requires the original immutable training plan")
    attempts = output / "resume_attempts"
    attempts.mkdir(exist_ok=True, mode=0o700)
    attempt = attempts / str(time.time_ns())
    attempt.mkdir(mode=0o700)
    proc = subprocess.run(
        _sparsify_command(plan),
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )
    (attempt / "native.stdout.log").write_text(proc.stdout)
    (attempt / "native.stderr.log").write_text(proc.stderr)
    (attempt / "exit.json").write_text(
        json.dumps({"returncode": proc.returncode}, sort_keys=True) + "\n"
    )
    if proc.returncode:
        raise RuntimeError("Original Sparsify checkpoint resume failed; inspect attempt logs")
    return {
        "status": "native_resume_returned_not_numerically_certified",
        "attempt_directory": str(attempt),
        "real_gpus_used": plan.distributed_gpus,
        "resume_equivalence_verified": False,
    }

def register_checkpoint(native_directory: Path, output: Path, *, source: dict) -> dict:
    """Hash an actual saved native checkpoint without copying third-party weights."""
    if output.exists():
        raise FileExistsError("Immutable checkpoint registry entry already exists")
    native_directory = native_directory.resolve()
    candidates = sorted(
        p
        for p in native_directory.rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".pt", ".pth", ".bin"}
    )
    if not candidates or len(candidates) > 1000:
        raise ValueError("Native checkpoint has no usable model tensor files")
    if not isinstance(source, dict) or not source.get("upstream_commit"):
        raise ValueError("Original native training commit/provenance is required")
    files = {}
    for file in candidates:
        h = hashlib.sha256()
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        files[str(file.relative_to(native_directory))] = h.hexdigest()
    record = {
        "schema": "opensae.wp12.native_checkpoint.v1",
        "native_source_directory": str(native_directory),
        "source": source,
        "tensor_file_sha256": files,
        "load_and_encode_verified": False,
        "resume_equivalence_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record
