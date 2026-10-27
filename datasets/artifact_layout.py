"""Canonical on-disk layout for SAEBase prompt artifacts (behavior × model)."""

from __future__ import annotations

from pathlib import Path

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


def _default_workspace_root() -> Path:
    """Use the source checkout for editable installs; cwd for installed wheels.

    Physical sources live in top-level datasets/ in a checkout, while installed
    distributions place them in saebase/datasets/ to avoid shadowing HF datasets.
    Writing experimental output into site-packages is never a useful default.
    """
    directory = Path(__file__).resolve().parent
    return Path.cwd().resolve() if directory.parent.name == "saebase" else directory.parent


# Prompt and experimental outputs share a workspace root, not a Python package root.
DEFAULT_DATA_ROOT = _default_workspace_root() / "data"
DEFAULT_EXPERIMENT_ROOT = _default_workspace_root() / "experiment"


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
