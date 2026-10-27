"""Shared NLP prompt identity, availability, schema and atomic artifact paths.

Consolidated verbatim behavior rules from SAEBase layout/schema/store; data
belongs to datasets rather than the SAE/model implementation package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SCHEMA_VERSION_PAIRED = 2
ARTIFACT_VERSION = "v1"

BEHAVIOR_DIRS: dict[str, str] = {
    "CounterFact": "CounterFact",
    "IOI": "IOI",
    "GreaterThan": "GreaterThan",
    "SubjectVerbAgreement": "SubjectVerbAgreement",
    "SimpleCopying": "SimpleCopying",
    "MMLUPro": "MMLUPro",
    "BBH": "BBH",
    "ARCChallenge": "ARCChallenge",
    "MedQA": "MedQA",
    "MATH500": "MATH500",
    "GPQAExtended": "GPQAExtended",
    "LiveCodeBench": "LiveCodeBench",
}

BEHAVIOR_ALIASES: dict[str, str] = {
    "counterfact": "CounterFact",
    "CounterFact": "CounterFact",
    "ioi": "IOI",
    "IOI": "IOI",
    "greater_than": "GreaterThan",
    "GreaterThan": "GreaterThan",
    "subject_verb_agreement": "SubjectVerbAgreement",
    "SubjectVerbAgreement": "SubjectVerbAgreement",
    "simple_copying": "SimpleCopying",
    "SimpleCopying": "SimpleCopying",
    "mmlu_pro": "MMLUPro",
    "MMLUPro": "MMLUPro",
    "MMLU-Pro": "MMLUPro",
    "bbh": "BBH",
    "BBH": "BBH",
    "arc_challenge": "ARCChallenge",
    "ARCChallenge": "ARCChallenge",
    "ARC-Challenge": "ARCChallenge",
    "medqa": "MedQA",
    "MedQA": "MedQA",
    "MedQA-USMLE": "MedQA",
    "math500": "MATH500",
    "MATH500": "MATH500",
    "MATH-500": "MATH500",
    "gpqa_extended": "GPQAExtended",
    "GPQAExtended": "GPQAExtended",
    "GPQA": "GPQAExtended",
    "livecodebench": "LiveCodeBench",
    "LiveCodeBench": "LiveCodeBench",
}

# Package-relative default: <SAEBase>/data
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
# Protocol / Hydra run artifacts (gitignore); not prompt artifacts.
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiment"


def canonicalize_behavior(name: str) -> str:
    key = str(name)
    if key not in BEHAVIOR_ALIASES:
        raise KeyError(f"Unknown behavior {key!r}. Valid: {sorted(set(BEHAVIOR_ALIASES.values()))}")
    return BEHAVIOR_ALIASES[key]


def default_data_root() -> Path:
    return DEFAULT_DATA_ROOT


def default_experiment_root() -> Path:
    """Hydra / protocol 运行产物根（gitignore）。不是 prompt artifact 的 ``data/``。"""
    return DEFAULT_EXPERIMENT_ROOT


def discovery_reports_root(data_root: str | Path | None = None) -> Path:
    """协议报告目录：默认 ``experiment/Discovery``；显式传入 ``data_root`` 时为 ``{data_root}/Discovery``。"""
    if data_root is not None:
        return Path(data_root) / "Discovery"
    return default_experiment_root() / "Discovery"


def artifact_dir(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
    version: str = ARTIFACT_VERSION,
) -> Path:
    """``{data_root}/{Behavior}/{model_key}/v1``."""
    beh = canonicalize_behavior(behavior)
    root = Path(data_root) if data_root is not None else default_data_root()
    return root / BEHAVIOR_DIRS[beh] / str(model_key) / version


def prompts_path(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
    version: str = ARTIFACT_VERSION,
) -> Path:
    return artifact_dir(behavior, model_key, data_root=data_root, version=version) / "prompts.jsonl"


def manifest_path(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
    version: str = ARTIFACT_VERSION,
) -> Path:
    return artifact_dir(behavior, model_key, data_root=data_root, version=version) / "manifest.json"


def balance_report_path(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
    version: str = ARTIFACT_VERSION,
) -> Path:
    return (
        artifact_dir(behavior, model_key, data_root=data_root, version=version)
        / "balance_report.json"
    )


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
    "gemma2_2b_base": _LOGIC_BEHAVIORS,
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


_ALLOWED_SCHEMA = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_PAIRED})


def save_prompts(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    data_root: str | Path | None = None,
) -> Path:
    behavior = canonicalize_behavior(str(manifest["behavior"]))
    model_key = str(manifest["model_key"])
    assert_available(behavior, model_key)
    schema_v = int(manifest.get("schema_version", -1))
    if schema_v not in _ALLOWED_SCHEMA:
        raise ValueError(
            f"manifest schema_version={manifest.get('schema_version')} "
            f"not in {sorted(_ALLOWED_SCHEMA)}"
        )
    if manifest.get("model_key") != model_key:
        raise ValueError("manifest.model_key mismatch")

    out_dir = artifact_dir(behavior, model_key, data_root=data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = prompts_path(behavior, model_key, data_root=data_root)
    man_file = manifest_path(behavior, model_key, data_root=data_root)

    with prompts_file.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    man_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Back-compat: callers historically expect the artifact directory Path.
    return out_dir


def load_manifest(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    beh = canonicalize_behavior(behavior)
    assert_available(beh, model_key)
    path = manifest_path(beh, model_key, data_root=data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt manifest not found for {beh}/{model_key}: {path}. "
            "Run saebase/tools/sae_prepare_prompts.py first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if str(manifest.get("model_key")) != str(model_key):
        raise RuntimeError(
            f"Artifact model_key mismatch: manifest={manifest.get('model_key')!r} "
            f"requested={model_key!r}"
        )
    if canonicalize_behavior(str(manifest.get("behavior", beh))) != beh:
        raise RuntimeError(
            f"Artifact behavior mismatch: manifest={manifest.get('behavior')!r} requested={beh!r}"
        )
    return manifest


def load_prompts(
    behavior: str,
    model_key: str,
    *,
    data_root: str | Path | None = None,
    require_manifest: bool = True,
) -> list[dict[str, Any]]:
    beh = canonicalize_behavior(behavior)
    if require_manifest:
        load_manifest(beh, model_key, data_root=data_root)
    path = prompts_path(beh, model_key, data_root=data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt artifact not found for {beh}/{model_key}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
