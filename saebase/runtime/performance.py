"""Bounded throughput tuning and durable task timing records.

Task adapters define work, units and numerical acceptance. This module measures
synchronized wall time, searches supported batches, and records actual evidence.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import math
import time

import numpy as np
import torch

_ACTIVE = ContextVar("saebase_performance", default=None)
_RECOMMENDATIONS = ContextVar("saebase_performance_recommendations", default=None)
_SHARED_POLICIES = ContextVar("saebase_loaded_pair_performance_policies", default=None)


@contextmanager
def policy_scope():
    """Reuse measured policies while the same loaded model/SAE lease is held."""
    recorder = _ACTIVE.get()
    policies = {
        "feature": getattr(recorder, "policies", {}),
        "answer": getattr(recorder, "answer_policies", {}),
    }
    if recorder is not None:
        recorder.policies = policies["feature"]
        recorder.answer_policies = policies["answer"]
    token = _SHARED_POLICIES.set(policies)
    try:
        yield
    finally:
        _SHARED_POLICIES.reset(token)


def remember_recommendations(records):
    """Transfer current precheck recommendations to its immediate task caller."""
    policies = {"feature": {}, "answer": {}}
    for row in records:
        for trial in row.get("optimizations", []):
            kind = trial.get("policy_kind")
            if "selected" in trial and kind in policies and trial.get("policy_key"):
                key = tuple(trial["policy_key"])
                recommendation = (
                    trial.get("runtime_batch_size", trial.get("selected_batch_size", 1))
                    if trial["selected"]
                    else 0
                )
                previous = policies[kind].get(key, recommendation)
                policies[kind][key] = min(previous, recommendation)
    _RECOMMENDATIONS.set(policies)


def consume_recommendations(recorder):
    """Apply a completed precheck to its owning runtime session exactly once."""
    recommendations = _RECOMMENDATIONS.get()
    _RECOMMENDATIONS.set(None)
    if recorder is not None and recommendations:
        for attribute, kind in [("policies", "feature"), ("answer_policies", "answer")]:
            policies = getattr(recorder, attribute, {})
            policies.update(recommendations[kind])
            setattr(recorder, attribute, policies)


def settings(cfg):
    raw = cfg.get("task", {}).get("performance", {})
    if not hasattr(raw, "get"):
        raise ValueError("task.performance must be a configuration mapping")
    result = {
        "enabled": raw.get("enabled", False),
        "batch_size": raw.get("batch_size", "auto"),
        "batch_sizes": list(raw.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64])),
        "warmup": raw.get("warmup", 1),
        "repeats": raw.get("repeats", 3),
        "probe_samples": raw.get("probe_samples", 3),
        "probe_states": raw.get("probe_states", 64),
        "max_seconds": raw.get("max_seconds", 120),
        "atol": raw.get("atol", 1e-6),
        "logit_atol": raw.get("logit_atol", 1e-4),
        "rtol": raw.get("rtol", 0.0),
        "memory_fraction": raw.get("memory_fraction", 0.9),
    }
    if type(result["enabled"]) is not bool:
        raise ValueError("task.performance.enabled must be boolean")
    for key in ["warmup", "repeats", "probe_samples", "probe_states"]:
        if type(result[key]) is not int or result[key] < (0 if key == "warmup" else 1):
            raise ValueError(f"task.performance.{key} has an invalid budget")
    if not result["batch_sizes"] or any(type(v) is not int or v < 1 for v in result["batch_sizes"]):
        raise ValueError("performance.batch_sizes must contain positive integers")
    if result["batch_size"] != "auto" and (
        type(result["batch_size"]) is not int or result["batch_size"] < 1
    ):
        raise ValueError("performance.batch_size must be auto or a positive integer")
    for key in ["max_seconds", "atol", "logit_atol", "rtol", "memory_fraction"]:
        if (
            not isinstance(result[key], (int, float))
            or not math.isfinite(result[key])
            or result[key] < 0
        ):
            raise ValueError(f"performance.{key} must be finite and nonnegative")
    if not 0 < result["memory_fraction"] <= 1:
        raise ValueError("performance.memory_fraction must be in (0,1]")
    from models.numerics import identity as numerical_identity

    if numerical_identity()["profile"] == "vllm":
        # One canonical arithmetic policy for scalar reference and batches.
        # Never hide a remaining batch effect by accepting a larger tolerance.
        result.update(atol=0.0, logit_atol=0.0, rtol=0.0)
    result["require_batch_invariance"] = numerical_identity()["profile"] == "vllm"
    return result


def cuda_measurement():
    active = _ACTIVE.get()
    return torch.cuda.is_available() and (
        active is None
        or str(active.device) in {"auto", "cuda"}
        or str(active.device).startswith("cuda:")
    )


def synchronize():
    if cuda_measurement():
        torch.cuda.synchronize()


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0}
    return {
        "count": len(x),
        "total_seconds": float(x.sum()),
        "mean_seconds": float(x.mean()),
        "std_seconds": float(x.std()),
        "min_seconds": float(x.min()),
        "max_seconds": float(x.max()),
        **{f"p{p}_seconds": float(np.percentile(x, p)) for p in [50, 90, 95, 99]},
    }


def timed(call):
    synchronize()
    started = time.perf_counter()
    result = call()
    synchronize()
    return result, time.perf_counter() - started


def tune(
    call,
    baseline,
    units,
    options,
    *,
    validate,
    candidate_setup_seconds=0.0,
    baseline_includes_setup=False,
):
    """Search measured batches, rejecting OOMs and changed scientific outputs.

    call(batch) must process the same complete probe at every batch size.
    Baseline is an independently measured unoptimized implementation.
    """
    overall_started = time.monotonic()
    for _ in range(options["warmup"]):
        baseline()
    baseline_times = []
    for _ in range(options["repeats"]):
        reference, seconds = timed(baseline)
        baseline_times.append(seconds)
    baseline_seconds = float(np.median(baseline_times))
    trials = []
    started = time.monotonic()
    candidates = sorted(
        v
        for v in set([1, min(units, max(options["batch_sizes"])), *options["batch_sizes"]])
        if v <= units
    )
    if options["batch_size"] != "auto":
        candidates = sorted({1, min(units, options["batch_size"])})
    for batch in candidates:
        if trials and time.monotonic() - started > options["max_seconds"]:
            break
        row = {"batch_size": batch, "units": units}
        try:
            if cuda_measurement():
                # Previous modes and larger candidates leave reusable allocator
                # blocks behind. Measure this candidate after releasing unused
                # blocks, so that history alone cannot reject every batch.
                row["reserved_before_cleanup_bytes"] = torch.cuda.memory_reserved()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                row["memory_measurement_policy"] = "per_candidate_after_unused_cache_release.v1"
            output, validation_seconds = timed(lambda: call(batch))
            error = validate(reference, output)
            row.update(validation_seconds=validation_seconds, **error)
            if not error["accepted"]:
                row["status"] = "numerically_rejected"
                if options.get("require_batch_invariance"):
                    raise ValueError(
                        f"Batch invariance failed at batch={batch}: {error}. "
                        "The performance precheck must be repaired before the production run; "
                        "a scalar fallback is not an accepted batch-invariant result."
                    )
            else:
                for _ in range(options["warmup"]):
                    call(batch)
                durations = [timed(lambda: call(batch))[1] for _ in range(options["repeats"])]
                row.update(
                    status="accepted",
                    timing=distribution(durations),
                    units_per_second=units
                    / (float(np.median(durations)) + candidate_setup_seconds),
                    kernel_units_per_second=units / float(np.median(durations)),
                )
            if cuda_measurement():
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                if row["peak_reserved_bytes"] > total * options["memory_fraction"]:
                    row["status"] = "memory_headroom_rejected"
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(
                exc, torch.cuda.OutOfMemoryError
            ):
                raise
            row.update(status="out_of_memory", reason=str(exc).splitlines()[0])
            if cuda_measurement():
                torch.cuda.empty_cache()
            trials.append(row)
            break
        trials.append(row)
    valid = [row for row in trials if row["status"] == "accepted"]
    best = max(valid, key=lambda row: row["units_per_second"]) if valid else None
    if (
        best is not None
        and options["batch_size"] == "auto"
        and best["units_per_second"] <= units / baseline_seconds
    ):
        best = None
    total_tuning_seconds = time.monotonic() - overall_started
    saved_per_unit = baseline_seconds / units - (
        1 / best["units_per_second"] if best else baseline_seconds / units
    )
    return {
        "baseline_seconds": baseline_seconds,
        "baseline_timing": distribution(baseline_times),
        "baseline_units_per_second": units / baseline_seconds,
        "selected_batch_size": best["batch_size"] if best else 1,
        "selected": best is not None,
        "trials": trials,
        "speedup": best["units_per_second"] * baseline_seconds / units if best else 1.0,
        "search_seconds": time.monotonic() - started,
        "total_tuning_seconds": total_tuning_seconds,
        "seconds_saved_per_unit": saved_per_unit,
        "break_even_units": total_tuning_seconds / saved_per_unit if saved_per_unit > 0 else None,
        "units": units,
        "baseline_includes_setup": baseline_includes_setup,
        "candidate_setup_seconds": candidate_setup_seconds,
        "comparison_scope": "Same probe; candidate throughput includes its measured one-time preparation",
        "search_scope": "best measured accepted candidate; not a global optimum",
    }


def numeric_validation(expected, actual, options):
    x, y = np.asarray(expected), np.asarray(actual)
    same = x.shape == y.shape and np.isfinite(y).all()
    return {
        "accepted": bool(same and np.allclose(x, y, atol=options["atol"], rtol=options["rtol"])),
        "max_abs_error": float(np.max(np.abs(x - y)))
        if same and x.size
        else (0.0 if same else None),
        "atol": options["atol"],
        "rtol": options["rtol"],
    }


class Recorder:
    def __init__(self, operation, root, device="auto"):
        self.operation, self.root, self.device = operation, Path(root), device
        self.phases, self.samples, self.experiments = {}, [], []
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name):
        synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize()
            self.phases.setdefault(name, []).append(time.perf_counter() - started)

    def sample(self, sample_id, seconds, *, unit="question", **extra):
        self.samples.append({"sample_id": sample_id, "seconds": seconds, "unit": unit, **extra})

    def save(self, error=None):
        from saebase.runtime.experiment_results import write_json
        from models.numerics import identity as numerical_identity
        from saebase.runtime.environment import worker_snapshot

        units = sorted({row["unit"] for row in self.samples})
        questions = [row for row in self.samples if row["unit"] == "question"]
        steady = []
        if len(questions) == len(self.experiments):
            for question, trial in zip(questions, self.experiments):
                if trial.get("total_tuning_seconds", 0) <= question["seconds"]:
                    steady.append(question["seconds"] - trial.get("total_tuning_seconds", 0))
        payload = {
            "schema": "saebase.task_performance.v1",
            "operation": self.operation,
            "wall_seconds": time.perf_counter() - self.started,
            "error": error,
            "phases": {key: distribution(value) for key, value in self.phases.items()},
            "per_sample": {
                unit: distribution([r["seconds"] for r in self.samples if r["unit"] == unit])
                for unit in units
            },
            "question_excluding_tuning": distribution(steady),
            "steady_timing_scope": "Question wall time minus measured batch-search work; mandatory cache preparation and numerical guards remain included",
            "observations": self.samples,
            "optimization_trials": self.experiments,
            "timing": "CUDA synchronized wall time; phase times may nest and must not be summed",
            "distribution_scope": "observed samples in this run; not a population confidence interval",
            "environment": {
                "numerics": numerical_identity(),
                "worker": worker_snapshot(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "torch_threads": torch.get_num_threads(),
                "device": torch.cuda.get_device_name() if cuda_measurement() else "cpu",
            },
        }
        write_json(self.root / "reports/performance.json", payload)
        lines = [
            f"# {self.operation} 性能报告",
            "",
            f"总墙钟时间：{payload['wall_seconds']:.3f} 秒。",
            "",
            "计时使用CUDA同步；阶段可能包含子阶段，不能直接相加。样本分布来自本次实际测量。",
            "",
            "| 单位 | 数量 | 平均秒 | P50 | P90 | P95 | P99 | 最小 | 最大 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for unit, row in payload["per_sample"].items():
            lines.append(
                "| "
                + unit
                + " | "
                + str(row["count"])
                + " | "
                + " | ".join(
                    f"{row[key]:.4f}"
                    for key in [
                        "mean_seconds",
                        "p50_seconds",
                        "p90_seconds",
                        "p95_seconds",
                        "p99_seconds",
                        "min_seconds",
                        "max_seconds",
                    ]
                )
                + " |"
            )
        lines += ["", "| 阶段 | 调用数 | 总秒数 | 平均秒数 |", "| --- | ---: | ---: | ---: |"]
        for name, row in payload["phases"].items():
            lines.append(
                f"| {name} | {row['count']} | {row['total_seconds']:.4f} | {row['mean_seconds']:.4f} |"
            )
        lines += [
            "",
            "优化试验及逐样本原始数据见同目录performance.json。收益仅表示相同输入、相同工作量的实测比值；未测量的优化不声称收益。",
        ]
        if steady:
            values = distribution(steady)
            lines += [
                "",
                f"扣除首次调优后的每题平均：{values['mean_seconds']:.4f} 秒；P50 {values['p50_seconds']:.4f}，P95 {values['p95_seconds']:.4f}。缓存准备、数值控制和写入仍计入。",
            ]
        (self.root / "reports/performance.md").write_text("\n".join(lines) + "\n")
        return payload


def current():
    return _ACTIVE.get()


@contextmanager
def session(cfg):
    if not settings(cfg)["enabled"] or not cfg.get("task", {}).get("execute", False):
        yield None
        return
    recorder = Recorder(str(cfg.task.name), Path.cwd(), str(cfg.task.get("device", "auto")))
    shared = _SHARED_POLICIES.get()
    if shared is not None:
        recorder.policies = shared["feature"]
        recorder.answer_policies = shared["answer"]
    consume_recommendations(recorder)
    token = _ACTIVE.set(recorder)
    error = None
    try:
        yield recorder
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        recorder.save(error)
        _ACTIVE.reset(token)


@contextmanager
def stage(name):
    active = current()
    if active is None:
        yield
    else:
        with active.stage(name):
            yield
