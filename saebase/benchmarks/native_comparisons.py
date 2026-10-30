"""P4 native-SAEBench pilot runner and provenance-aware SAE comparison.

Reuses original upstream Core/Sparse-Probing code; native JSON import is
not a substitute for running the upstream evaluators on real checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

SHA = re.compile(r"^[0-9a-f]{40}$")
NATIVE_PIN = "8042bb3828c6340da8d12062324e92b2077c571c"


@dataclass(frozen=True)
class NativePilot:
    repo_root: Path
    upstream: Path
    output: Path
    upstream_commit: str = NATIVE_PIN
    timeout_seconds: int = 1800

    def validate(self) -> None:
        if not SHA.fullmatch(self.upstream_commit):
            raise ValueError("Require the exact upstream commit SHA")
        if not 30 <= self.timeout_seconds <= 7200:
            raise ValueError("Bounded C1/C2 timeout required")
        if not (self.repo_root / "saebase/tools/m3_native_saebench_smoke.py").is_file():
            raise FileNotFoundError("Missing SAEBase native SAEBench entrypoint")
        expected = self.repo_root / "upstream"
        if self.upstream.resolve() != expected.resolve():
            raise ValueError("Pinned native runner requires repo_root/upstream checkout")
        commit = subprocess.check_output(
            ["git", "-C", str(self.upstream), "rev-parse", "HEAD"],
            text=True,
            timeout=20,
        ).strip()
        if commit != self.upstream_commit:
            raise ValueError("Native upstream checkout does not match frozen commit")
        if self.output.exists():
            raise FileExistsError("Never overwrite native benchmark run or raw evidence")


def native_pilot(job: NativePilot) -> dict[str, Any]:
    """Execute the original upstream benchmark smoke, not fabricated metrics."""
    job.validate()
    job.output.mkdir(mode=0o700, parents=True)
    env = {
        **os.environ,
        "SAEBASE_NATIVE_RESULTS": str(job.output.resolve()),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    proc = subprocess.run(
        [sys.executable, "saebase/tools/m3_native_saebench_smoke.py"],
        cwd=job.repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=job.timeout_seconds,
        check=False,
    )
    (job.output / "stdout.log").write_text(proc.stdout)
    (job.output / "stderr.log").write_text(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"Native pilot failed with return code {proc.returncode}; inspect logs")
    path = job.output / "evidence.json"
    if not path.is_file():
        raise RuntimeError("Original evaluator produced no evidence.json")
    record = json.loads(path.read_text())
    if (
        record.get("status") != "passed"
        or record.get("upstream_commit") != job.upstream_commit
        or set(record.get("native_metrics", {}))
        != {
            "saebench.core.l0",
            "saebench.core.mse",
            "saebench.sparse_probing.sae_top_1_test_accuracy",
        }
    ):
        raise RuntimeError("Upstream native results or source identity are invalid")
    return {
        "status": "native_core_sparse_probing_pilot_completed",
        "upstream_commit": job.upstream_commit,
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_result_files": [record["core_file"], record["sparse_probing_file"]],
        "metrics": record["native_metrics"],
        "native_execution": True,
    }


@dataclass(frozen=True)
class ComparisonPoint:
    """An actual native benchmark metric row; labels are not inferred."""

    model_sha256: str
    sae_sha256: str
    hook: str
    dataset_sha256: str
    native_commit: str
    metric_id: str
    metric_value: float
    l0: float
    sae_name: str

    def __post_init__(self) -> None:
        for name in ("model_sha256", "sae_sha256", "dataset_sha256"):
            v = getattr(self, name)
            if not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v):
                raise ValueError("Comparison requires full source weight/dataset digests")
        if not SHA.fullmatch(self.native_commit):
            raise ValueError("Native upstream commit must be pinned")
        if not self.hook or not self.metric_id or not self.sae_name:
            raise ValueError("Complete site, metric and SAE identities required")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (self.metric_value, self.l0)
            )
            or self.l0 < 0
        ):
            raise ValueError("Metrics and L0 must be finite")


def compare_sae_points(points: list[ComparisonPoint]) -> dict[str, Any]:
    """Produce a *fairness-checked* Pareto series, never invented metrics."""
    if len(points) < 2:
        raise ValueError("Compare at least two genuinely evaluated SAEs")
    ref = points[0]
    for p in points[1:]:
        mismatches = [
            name
            for name in ("model_sha256", "hook", "dataset_sha256", "native_commit", "metric_id")
            if getattr(p, name) != getattr(ref, name)
        ]
        if mismatches:
            return {
                "comparable": False,
                "mismatches": mismatches,
                "points": [],
                "reason": "Source/checkpoint/site/protocol are not matched",
            }
    if len({p.sae_sha256 for p in points}) < 2:
        raise ValueError("Two distinct SAE checkpoints are required")
    return {
        "comparable": True,
        "metric": ref.metric_id,
        "scope": {
            "model_sha256": ref.model_sha256,
            "hook": ref.hook,
            "dataset_sha256": ref.dataset_sha256,
            "native_commit": ref.native_commit,
        },
        "points": [
            {
                "sae": p.sae_name,
                "sae_sha256": p.sae_sha256,
                "l0": p.l0,
                "metric_value": p.metric_value,
            }
            for p in sorted(points, key=lambda q: (q.l0, q.sae_name))
        ],
        "uncertainty": "not_computed_without_independent_sample_values",
    }
