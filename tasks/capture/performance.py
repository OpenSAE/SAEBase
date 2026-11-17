"""Numerically checked, adaptive batching of canonical-answer edits.

Keeps the original native full-sequence scalar scorer as the acceptance oracle.
Cache and batch candidates are measured independently, with per-prompt guards.
"""

import time
import numpy as np
import torch

from saebase.api import ActivationSite
from saebase.api.replay import SiteReplay
from saebase.runtime.performance import tune, numeric_validation, current, stage


class AnswerReplay:
    def __init__(self, pair, snapshot, answer_ids, readout_activation, prepared_hidden=None):
        self.snapshot, self.reference = snapshot, readout_activation
        self.answer = torch.tensor(answer_ids, dtype=torch.long, device=readout_activation.device)
        self.position = snapshot.tokens.shape[1] - 1
        tokens = torch.cat(
            [snapshot.tokens.to(readout_activation.device), self.answer[None, :-1]], 1
        )
        self.replay = SiteReplay(
            pair.model,
            tokens,
            ActivationSite(snapshot.site.layer, snapshot.site.hook_name, self.position),
            prepared_hidden=prepared_hidden,
        )

    @torch.no_grad()
    def __call__(self, activations, batch_size):
        output = []
        for start in range(0, len(activations), batch_size):
            targets = activations[start : start + batch_size].to(self.reference.device)
            targets = (
                self.reference.float()
                + (targets.float() - self.snapshot.activation.to(self.reference.device).float())
            ).to(self.reference.dtype)
            logits = self.replay.logits(targets)[:, self.position :]
            # Match the reference float32 normalization and float64 token mean.
            logp = logits.float().log_softmax(-1)
            selected = logp.gather(
                2, self.answer[None, :, None].expand(len(targets), -1, -1)
            ).squeeze(-1)
            output.extend(selected.double().mean(-1).cpu().tolist())
            del logits, logp, selected
        return np.asarray(output, dtype=np.float64)


def answer_policy_key(pair, snapshot, answer_ids, control_count):
    """Bucket total replay length, retaining small-vs-large workload differences."""
    length = snapshot.tokens.shape[1] + len(answer_ids) - 1
    return (
        pair.identity.model_key,
        snapshot.site.hook_name,
        1 << (length - 1).bit_length(),
        control_count,
        "answer-shape-v2",
    )


def evaluate_edits(pair, snapshot, activations, answer_ids, readout_activation, native, options):
    started = time.perf_counter()
    with stage("suffix_cache_prepare"):
        candidate = AnswerReplay(
            pair, snapshot, answer_ids, readout_activation, getattr(native, "prepared_hidden", None)
        )
    setup_seconds = time.perf_counter() - started
    count = min(len(activations), options["probe_states"])
    # Evenly spread controls cover all-kept/all-deleted and intermediate masks.
    indices = np.unique(np.linspace(0, len(activations) - 1, count, dtype=int))
    control = activations[indices]

    def baseline():
        return np.asarray([float(native(value)) for value in control])

    recorder = current()
    policies = getattr(recorder, "answer_policies", {}) if recorder is not None else {}
    key = answer_policy_key(pair, snapshot, answer_ids, len(control))
    tuning = None
    if key in policies and policies[key] == 0:
        tuning = {
            "selected": False,
            "selected_batch_size": 1,
            "policy_reused": True,
            "reason": "The original scalar scorer was the best measured accepted option for this bucket",
        }
    elif key in policies:
        batch = policies[key]
        with stage("per_prompt_numerical_guard"):
            expected = np.asarray([float(native(value)) for value in control[:4]])
            actual = candidate(control[:4], batch)
            guard = numeric_validation(expected, actual, options)
        if guard["accepted"]:
            tuning = {
                "selected": True,
                "selected_batch_size": batch,
                "policy_reused": True,
                "validation": guard,
            }
    if tuning is None:
        with stage("autotune"):
            tuning = tune(
                lambda batch: candidate(control, batch),
                baseline,
                len(control),
                options,
                validate=lambda a, b: numeric_validation(a, b, options),
                candidate_setup_seconds=setup_seconds,
            )
        policies[key] = tuning["selected_batch_size"] if tuning["selected"] else 0
        if recorder is not None:
            recorder.answer_policies = policies
    tuning.update(
        optimization="cached_suffix_and_state_batching",
        cache_execution=candidate.replay.execution,
        cache_setup_seconds=setup_seconds,
        state_unit="canonical_answer_forward",
        policy_kind="answer",
        policy_key=list(key),
        interaction_absolute_error_budget_on_controls=4 * options["atol"],
        validation_scope="sampled control states, not a numerical bound on untested states",
    )
    if recorder is not None:
        recorder.experiments.append(tuning)
    forward_started = time.perf_counter()
    with stage("answer_forwards"):
        if not tuning["selected"]:
            values = np.asarray([float(native(value)) for value in activations])
        else:
            batch = tuning["selected_batch_size"]
            while True:
                try:
                    values = candidate(activations, batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch == 1:
                        raise
                    batch = max(1, batch // 2)
                    torch.cuda.empty_cache()
                    tuning.setdefault("runtime_oom_batches", []).append(batch)
                    expected = np.asarray([float(native(value)) for value in control[:4]])
                    guard = numeric_validation(expected, candidate(control[:4], batch), options)
                    if not guard["accepted"]:
                        if options.get("require_batch_invariance"):
                            raise ValueError(
                                f"Batch invariance failed after reducing an OOM batch: {guard}"
                            )
                        tuning["runtime_fallback"] = (
                            "OOM fallback batch failed the numerical guard; used the original scalar scorer"
                        )
                        values = np.asarray([float(native(value)) for value in activations])
                        batch = 1
                        break
            tuning["runtime_batch_size"] = batch
            policies[key] = batch
    tuning["runtime_seconds"] = time.perf_counter() - forward_started
    tuning["seconds_per_unique_state"] = tuning["runtime_seconds"] / len(activations)
    if recorder is not None:
        recorder.sample(
            str(len(recorder.samples)),
            tuning["seconds_per_unique_state"],
            unit="answer_state_amortized",
            unique_states=len(activations),
            timing_scope="full answer-forward phase divided by unique native-code states",
        )
    return values, tuning


@torch.no_grad()
def prepare_activations(sae, snapshot, ids, masks):
    """Reuse only the unchanged native decoder baseline and identical activations.

    Decode each distinct edited code separately, preserving native GEMM shapes,
    arithmetic ordering and snapshot layer-normalization statistics.
    """
    from saebase.api.features import _decode

    kwargs = {"ln_mu": snapshot.ln_mu, "ln_std": snapshot.ln_std}
    before = _decode(sae, snapshot.codes, snapshot.activation.numel(), **kwargs)
    values = []
    inverse = []
    seen = {}
    active_ids = {
        fid for fid, value in zip(ids, snapshot.codes[ids].detach().cpu().tolist()) if value != 0
    }
    for mask in masks:
        removed = [fid for fid, bit in zip(ids, mask) if not bit and fid in active_ids]
        key = tuple(removed)
        if key not in seen:
            if removed:
                codes = snapshot.codes.clone()
                codes[removed] = 0
                after = _decode(sae, codes, snapshot.activation.numel(), **kwargs)
                delta = (after - before).to(snapshot.activation)
                value = snapshot.activation + delta
            else:
                value = snapshot.activation
            seen[key] = len(values)
            values.append(value)
        inverse.append(seen[key])
    return (
        torch.stack(values),
        np.asarray(inverse),
        {
            "original_states": len(masks),
            "unique_native_code_states": len(values),
            "optimizations": [
                "cached_native_decoder_baseline",
                "identical_code_state_deduplication",
            ],
            "deduplication_forward_reduction": len(masks) / len(values),
        },
    )


@torch.no_grad()
def verify_batch_artifact(pair, artifact, *, batch_size=16):
    """Re-read all I responses and independently batch them while the pair is loaded."""
    from pathlib import Path
    from saebase.api import capture_features
    from saebase.runtime.performance import synchronize

    synchronize()
    started = time.perf_counter()
    device = next(pair.model.model.parameters()).device
    dtype = next(pair.model.model.parameters()).dtype
    with np.load(artifact) as data:
        snapshot = capture_features(
            pair.model,
            pair.sae,
            site=pair.site,
            tokens=torch.tensor(data["source_tokens"], device=device),
        )
        activations, inverse, _ = prepare_activations(
            pair.sae, snapshot, data["feature_ids"].tolist(), data["masks"]
        )
        candidate = AnswerReplay(
            pair,
            snapshot,
            data["answer_token_ids"].tolist(),
            torch.tensor(data["readout_activation"], device=device, dtype=dtype),
        )
        actual = candidate(activations, batch_size)[inverse]
        expected = data["responses"]
        error = float(np.max(np.abs(expected - actual)))
        if not np.array_equal(expected, actual):
            raise ValueError(
                f"Complete I artifact is not batch-invariant: {artifact}, max error={error}"
            )
        count = len(expected)
    synchronize()
    return {
        "status": "PASS",
        "artifact": str(Path(artifact)),
        "states_checked": count,
        "unique_states": len(activations),
        "additional_batch_size": batch_size,
        "max_abs_error": error,
        "seconds": time.perf_counter() - started,
    }
