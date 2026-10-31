"""Read one pinned Hugging Face dataset without choosing a benchmark cohort."""

from __future__ import annotations

from typing import Any, Iterator

from saebase.data.qa_records import DatasetSpec


def iter_source(
    spec: DatasetSpec, revision: str
) -> Iterator[tuple[str | None, int, dict[str, Any]]]:
    from datasets import get_dataset_config_names, load_dataset

    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Dataset source revision must be a full 40-char commit")
    configs = (
        sorted(get_dataset_config_names(spec.repo, revision=revision))
        if spec.kind == "bbh"
        else [spec.config]
    )
    if spec.kind == "bbh" and len(configs) < 20:
        raise RuntimeError("BBH should expose multiple real task configurations")
    for config in configs:
        dataset = load_dataset(spec.repo, name=config, split=spec.split, revision=revision)
        for index, row in enumerate(dataset):
            yield config, index, dict(row)
