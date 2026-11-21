"""Locate measurements and exports from the historical fifty-question SAR run.

The active SAR scripts use this module for per-question reads and resumable writes.
The flat lookup keeps earlier standalone copies of the measurements readable.
"""

from pathlib import Path


RESULTS = Path(__file__).resolve().parents[3] / "assets/history_experiment/response_surface/files"
_DATASET_DIRECTORIES = {
    "ARCChallenge": "arc_challenge",
    "BBH": "bbh",
    "GSM8K": "gsm8k",
    "MMLUPro": "mmlu_pro",
    "TruthfulQA": "truthful_qa",
}


def question_directory(root: Path, key: str) -> Path:
    dataset, _, number = key.rpartition("-")
    if dataset not in _DATASET_DIRECTORIES or not number.isascii() or not number.isdecimal():
        raise ValueError(f"Invalid SAR question key: {key!r}")
    grouped = root / "by_dataset" / _DATASET_DIRECTORIES[dataset] / key
    flat = root / key
    return flat if flat.is_dir() and not grouped.exists() else grouped
