"""Behavior × model availability and shared prompt record helpers."""

from __future__ import annotations

from typing import Any

from saebase.datasets.artifact_layout import canonicalize_behavior

_LOGIC_BEHAVIORS: frozenset[str] = frozenset(
    {
        "CounterFact",
        "IOI",
        "GreaterThan",
        "SubjectVerbAgreement",
        "SimpleCopying",
        "MMLUPro",
        "BBH",
    }
)

_REASONING_BENCHS: frozenset[str] = frozenset(
    {
        "ARCChallenge",
        "MedQA",
        "MATH500",
        "GPQAExtended",
        "LiveCodeBench",
    }
)

_EXP10_BEHAVIORS: frozenset[str] = _LOGIC_BEHAVIORS | _REASONING_BENCHS

# Experiment 11b targeted set: keep existing logic benches + MedQA/MATH-500 only.
_11B_TARGET_EXTRA: frozenset[str] = frozenset({"MedQA", "MATH500"})
_11B_BEHAVIORS: frozenset[str] = _LOGIC_BEHAVIORS | _11B_TARGET_EXTRA

# Central capability matrix (do not import from model modules).
MODEL_BEHAVIOR_MATRIX: dict[str, frozenset[str]] = {
    "gpt2small": frozenset({"CounterFact", "SimpleCopying", "SubjectVerbAgreement"}),
    "pythia70m": frozenset({"CounterFact", "SimpleCopying", "SubjectVerbAgreement"}),
    "pythia410m": frozenset({"SimpleCopying", "SubjectVerbAgreement"}),
    "gemma270m": frozenset({"SimpleCopying", "SubjectVerbAgreement"}),
    "gemma1b": _LOGIC_BEHAVIORS,
    "gemma1b_it": _LOGIC_BEHAVIORS,
    "r1_distill_qwen15b": _LOGIC_BEHAVIORS,
    "r1_distill_llama8b": _11B_BEHAVIORS,
    "gemma2_2b_base": _LOGIC_BEHAVIORS,
    "gemma2_2b_it": _11B_BEHAVIORS,
    "qwen25_3b": _LOGIC_BEHAVIORS,
    "qwen25_3b_it": _LOGIC_BEHAVIORS,
    "gemma3_4b_pt": _LOGIC_BEHAVIORS,
    "gemma3_4b_it": _LOGIC_BEHAVIORS,
    "llama31_8b": _LOGIC_BEHAVIORS,
    "llama31_8b_instruct": _EXP10_BEHAVIORS,
    "qwen3_8b": _LOGIC_BEHAVIORS,
    "qwen3_1_7b": _11B_BEHAVIORS,
    "qwen3_1_7b_base": _LOGIC_BEHAVIORS,
    "mistral7b_instruct": _LOGIC_BEHAVIORS,
    "qwen35_2b": _LOGIC_BEHAVIORS,
    "qwen35_4b": _11B_BEHAVIORS,
    "gemma2_9b_pt": _LOGIC_BEHAVIORS,
    "gemma2_9b_it": _EXP10_BEHAVIORS,
    "gemma3_12b_pt": _LOGIC_BEHAVIORS,
    "gemma3_12b_it": _LOGIC_BEHAVIORS,
    "qwen35_9b": _LOGIC_BEHAVIORS,
    "gemma3_27b_pt": _LOGIC_BEHAVIORS,
    "gemma3_27b_it": _LOGIC_BEHAVIORS,
    "qwen35_27b": _LOGIC_BEHAVIORS,
}

TEXT_BEHAVIORS: frozenset[str] = _LOGIC_BEHAVIORS | _REASONING_BENCHS


def behaviors_for_model(model_key: str) -> frozenset[str]:
    return MODEL_BEHAVIOR_MATRIX[str(model_key)]


def is_available(behavior: str, model_key: str) -> bool:
    beh = canonicalize_behavior(behavior)
    allowed = MODEL_BEHAVIOR_MATRIX.get(str(model_key))
    if allowed is None:
        return False
    return beh in allowed


def assert_available(behavior: str, model_key: str) -> None:
    beh = canonicalize_behavior(behavior)
    key = str(model_key)
    if key not in MODEL_BEHAVIOR_MATRIX:
        raise KeyError(f"Unknown model_key={key!r}. Valid: {sorted(MODEL_BEHAVIOR_MATRIX)}")
    if beh not in MODEL_BEHAVIOR_MATRIX[key]:
        raise RuntimeError(
            f"Behavior {beh!r} is not available for model_key={key!r}. "
            f"Supported: {sorted(MODEL_BEHAVIOR_MATRIX[key])}"
        )


def require_keys(record: dict[str, Any], keys: list[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in record]
    if missing:
        raise ValueError(f"{ctx}: missing keys {missing}")


def make_manifest(
    *,
    behavior: str,
    model_key: str,
    model_name: str,
    loaded_model_name: str,
    seed: int,
    generated_count: int,
    kept_count: int,
    filter_policy: dict[str, Any],
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from saebase.datasets.artifact_layout import SCHEMA_VERSION

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "behavior": canonicalize_behavior(behavior),
        "model_key": str(model_key),
        "model_name": model_name,
        "loaded_model_name": loaded_model_name,
        "tokenizer_name": loaded_model_name,
        "seed": int(seed),
        "generated_count": int(generated_count),
        "kept_count": int(kept_count),
        "filter_policy": filter_policy,
        "source": source,
    }
    if extra:
        payload.update(extra)
    return payload
