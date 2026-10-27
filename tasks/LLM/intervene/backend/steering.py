"""Reference token-by-token generation with per-step SAE steering.

Each generated step recomputes the entire prefix; no KV cache assumptions.
This slow, unambiguous reference is an acceptance oracle for future high-speed
vLLM/NNsight or cached-decoding implementations, not a performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features
from saebase.store import sha256_state_dict, fingerprint
from .operators import apply_condition
from .results import ResultWriter
from .specs import Condition, ExperimentIdentity, MeasurementSample
from .engine import RunBudget


@dataclass(frozen=True)
class GenerationSpec:
    max_new_tokens: int = 8
    seed: int = 0
    temperature: float = 0.0  # 0 is greedy; positive uses categorical sampling.
    top_k: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if (
            not isinstance(self.temperature, (float, int))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be nonnegative and finite")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer")
        if any(type(i) is not int or i < 0 for i in self.eos_token_ids) or len(
            set(self.eos_token_ids)
        ) != len(self.eos_token_ids):
            raise ValueError("eos_token_ids must be distinct nonnegative token IDs")
        if self.max_context_tokens is not None and (
            type(self.max_context_tokens) is not int or self.max_context_tokens < 1
        ):
            raise ValueError("max_context_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "temperature": float(self.temperature),
            "top_k": self.top_k,
            "eos_token_ids": list(self.eos_token_ids),
            "max_context_tokens": self.max_context_tokens,
        }


def _distribution_shift(
    clean: torch.Tensor, edited: torch.Tensor, *, feature_id: int | None, activation: float | None
) -> dict[str, Any]:
    """Teacher-forced next-token shift at one generated prefix. Not a continuation."""
    clean_f = clean.detach().float().cpu()
    edited_f = edited.detach().float().cpu()
    if clean_f.ndim != 1 or edited_f.shape != clean_f.shape:
        raise RuntimeError("generation logits must be one shared vocabulary vector")
    clean_prob = torch.softmax(clean_f, -1)
    edited_prob = torch.softmax(edited_f, -1)
    clean_top = int(clean_f.argmax().item())
    edited_top = int(edited_f.argmax().item())
    rank = int((torch.argsort(edited_f, descending=True) == clean_top).nonzero()[0].item())
    return {
        "feature_id": feature_id,
        "feature_activation": activation,
        "clean_argmax_token_id": clean_top,
        "edited_argmax_token_id": edited_top,
        "clean_argmax_probability": float(clean_prob[clean_top].item()),
        "edited_probability_of_clean_argmax": float(edited_prob[clean_top].item()),
        "edited_argmax_probability": float(edited_prob[edited_top].item()),
        "clean_argmax_rank_after_edit": rank,
        "max_abs_logit_delta": float((edited_f - clean_f).abs().max().item()),
        "logit_kind": "teacher_forced_next_token_at_generation_step",
    }


def _sample(logits: torch.Tensor, spec: GenerationSpec, *, rng: torch.Generator) -> int:
    vector = logits.float().detach().cpu()
    if not bool(torch.isfinite(vector).all()):
        raise RuntimeError("generation logits contain non-finite values")
    if spec.temperature == 0:
        return int(torch.argmax(vector).item())
    if spec.top_k is not None:
        k = min(spec.top_k, vector.numel())
        values, indices = torch.topk(vector, k=k)
        probs = torch.softmax(values / spec.temperature, -1)
        return int(indices[torch.multinomial(probs, 1, generator=rng)[0]].item())
    probs = torch.softmax(vector / spec.temperature, -1)
    return int(torch.multinomial(probs, 1, generator=rng)[0].item())


@torch.no_grad()
def compare_steered_generation(
    model: ModelAdapter,
    sae: Any,
    *,
    site: ActivationSite,
    sample: MeasurementSample,
    condition: Condition,
    spec: GenerationSpec,
) -> dict[str, Any]:
    """Generate independent baseline and steered continuations from the same prompt.

    Baseline and steered traces use identically seeded independent RNGs (common
    random numbers at each generation step). Each trace's own generated prefix
    determines its next token; later patches never reuse stale KV cache state.
    """
    model.capabilities.require("forward")
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    original = sample.tokens
    if (
        spec.max_context_tokens is not None
        and len(original) + spec.max_new_tokens > spec.max_context_tokens
    ):
        raise ValueError("requested generation exceeds the configured context limit")
    eos = set(spec.eos_token_ids)
    result: dict[str, Any] = {}
    for label, steered in (("baseline", False), ("steered", True)):
        seed_material = hashlib.sha256(f"{spec.seed}:{sample.record.key}".encode()).digest()
        rng = torch.Generator(device="cpu").manual_seed(
            int.from_bytes(seed_material[:8], "little") % (2**63 - 1)
        )
        current = list(original)
        generated: list[int] = []
        steps: list[dict[str, Any]] = []
        for step in range(spec.max_new_tokens):
            input_ids = model.tokens(tokens=torch.tensor([current], dtype=torch.long))
            if steered:
                position = len(current) - 1
                current_site = ActivationSite(site.layer, site.hook_name, position)
                snapshot = capture_features(model, sae, site=current_site, tokens=input_ids)
                patch = apply_condition(sae, snapshot, condition)
                logits = model.patch_logits(
                    tokens=input_ids,
                    site=current_site,
                    activation=patch.activation,
                    readout_position=-1,
                )
                clean = model.forward_logits(tokens=input_ids, readout_position=-1)
                feature_id = condition.feature_ids[0] if condition.feature_ids else None
                activation = None
                if feature_id is not None:
                    activation = float(snapshot.codes[feature_id].detach().float().item())
                effect = _distribution_shift(
                    clean, logits, feature_id=feature_id, activation=activation
                )
                effect["step"] = step
                effect["position"] = position
                steps.append(effect)
            else:
                logits = model.forward_logits(tokens=input_ids, readout_position=-1)
            tok = _sample(logits, spec, rng=rng)
            generated.append(tok)
            current.append(tok)
            if tok in eos:
                break
        result[label + "_generated_token_ids"] = generated
        result[label + "_stop_reason"] = (
            "eos" if generated and generated[-1] in eos else "max_new_tokens"
        )
        if steered:
            result["steered_steps"] = steps
    result["same_continuation"] = (
        result["baseline_generated_token_ids"] == result["steered_generated_token_ids"]
    )
    result["shared_prefix_tokens"] = len(original)
    result["first_different_generated_index"] = next(
        (
            i
            for i, (a, b) in enumerate(
                zip(result["baseline_generated_token_ids"], result["steered_generated_token_ids"])
            )
            if a != b
        ),
        None,
    )
    return result


@torch.no_grad()
def run_steering(
    model: ModelAdapter,
    sae: Any,
    identity: ExperimentIdentity,
    samples: Sequence[MeasurementSample],
    condition: Condition,
    spec: GenerationSpec,
    writer: ResultWriter,
    *,
    budget: RunBudget | None = None,
) -> dict[str, Any]:
    if writer.identity.key != identity.key:
        raise ValueError("result writer identity mismatch")
    if (
        model.backend != identity.backend
        or sha256_state_dict(model.model) != identity.model_state_sha256
        or sha256_state_dict(sae) != identity.sae_state_sha256
    ):
        raise ValueError("model/SAE differs from recorded identity")
    limits = budget or RunBudget()
    if (
        len(samples) > limits.max_measurements
        or 2 * len(samples) * spec.max_new_tokens > limits.max_generated_tokens
    ):
        raise ValueError("generation exceeds declared token/sample budget")
    done, reused = 0, 0
    for sample in samples:
        measurement_identity = {
            "schema": "saebase.m2.steering.v1",
            "experiment": identity.key,
            "sample": sample.to_dict(),
            "condition": condition.to_dict(),
            "generation": spec.to_dict(),
        }
        key = fingerprint(measurement_identity)
        if writer.has(key):
            reused += 1
            continue
        value = compare_steered_generation(
            model, sae, site=identity.site, sample=sample, condition=condition, spec=spec
        )
        if writer.append(
            {
                "schema": "saebase.m2.steering.v1",
                "measurement_identity": measurement_identity,
                "measurement_id": key,
                "sample_id": sample.record.sample_id,
                "condition_name": condition.name,
                "generation": spec.to_dict(),
                "result": value,
            }
        ):
            done += 1
        writer.flush()
    return {
        "measured": done,
        "reused": reused,
        "tokens_budgeted": 2 * len(samples) * spec.max_new_tokens,
    }
