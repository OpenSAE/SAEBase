"""One Dataset-owned fixed cohort, shared by precheck and formal Tasks.

Natural cohorts sample the union of original splits. Split labels are provenance,
not selection criteria. Synthetic owners must produce exactly the requested size.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from saebase.datasets.artifacts import _in_prepared_assets, resolve_prepared_path


@dataclass
class SampleSelection:
    records: Iterable[dict]
    metadata: dict

    def __iter__(self):
        return iter(self.records)


def validate_selection(cfg: Any) -> None:
    for key in ("sample_count", "sampling_seed"):
        value = cfg.get(key, 500 if key == "sample_count" else 42)
        if type(value) is not int or value < (1 if key == "sample_count" else 0):
            raise ValueError(f"dataset.{key} must be a valid integer")
    if any(key in cfg for key in ("max_samples", "min_samples", "purpose", "split")):
        raise ValueError(
            "Dataset cohort is controlled by sample_count and sampling_seed; split is retained on records only"
        )
    if str(cfg.get("name", "")) == "IOI":
        maximum = cfg.get("max_candidates", 100000)
        if type(maximum) is not int or maximum < cfg.get("sample_count", 500):
            raise ValueError("dataset.max_candidates must cover the requested synthetic cohort")
    if type(cfg.get("use_original_split", False)) is not bool:
        raise ValueError("dataset.use_original_split must be boolean")


def selection_path(cfg, model_key):
    validate_selection(cfg)
    specification = {
        "schema": "saebase.dataset.fixed_cohort.v1",
        "dataset": str(cfg.name),
        "count": cfg.get("sample_count", 500),
        "seed": cfg.get("sampling_seed", 42),
        "source_config": cfg.get("source_config"),
        "source_revision": cfg.get("source_revision"),
        "prompt_file": cfg.get("prompt_file") or None,
    }
    if cfg.get("prompt_file"):
        from hydra.utils import to_absolute_path

        source = Path(to_absolute_path(str(cfg.prompt_file))).expanduser().resolve()
        specification["prompt_file"] = str(source)
        specification["source_sha256"] = hashlib.sha256(resolve_prepared_path(source).read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:24]
    root = resolve_prepared_path(Path(str(cfg.data_path)))
    if _in_prepared_assets(root):
        owner = model_key if str(cfg.name) == "IOI" else "shared"
        return root / owner / "selections" / f"cohort_{key}"
    if str(cfg.name) == "IOI":
        return root / model_key / "selections" / key
    selection = root / "shared" / "selections" / key
    legacy = root / "_shared" / "selections" / key
    if selection.exists():
        return selection
    if legacy.exists() or root.parts[-4:-1] != ("data", "prompts", "prepared"):
        return legacy
    return selection


def read_selection(cfg, model_key):
    path = selection_path(cfg, model_key)
    data = (path / "records.jsonl").read_bytes()
    metadata = json.loads((path / "manifest.json").read_text())
    if metadata.get("schema") != "saebase.dataset.fixed_cohort.v1":
        raise ValueError("Unknown Dataset cohort format")
    if hashlib.sha256(data).hexdigest() != metadata["records_sha256"]:
        raise ValueError("Dataset cohort checksum mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if (
        not rows
        or len(rows) > cfg.get("sample_count", 500)
        or len(rows) != metadata["selected_count"]
        or len({r["sample_id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Dataset cohort count or identity mismatch")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise ValueError("Dataset cohort contains an invalid prompt")
    if (
        hashlib.sha256(json.dumps([row["sample_id"] for row in rows]).encode()).hexdigest()
        != metadata["sample_ids_sha256"]
    ):
        raise ValueError("Dataset cohort sample identity digest mismatch")
    if (
        metadata["dataset"] != str(cfg.name)
        or metadata["requested_count"] != cfg.get("sample_count", 500)
        or metadata["sampling_seed"] != cfg.get("sampling_seed", 42)
    ):
        raise ValueError("Dataset cohort configuration mismatch")
    if str(cfg.name) == "IOI" and len(rows) != cfg.get("sample_count", 500):
        raise ValueError("Synthetic Dataset must contain exactly sample_count eligible records")
    return SampleSelection(rows, {**metadata, "artifact_path": str(path)})


def save_selection(cfg, model_key, rows, metadata):
    from saebase.runtime.experiment_results import write_json

    path = selection_path(cfg, model_key)
    path.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    temporary = path / "records.jsonl.tmp"
    temporary.write_bytes(data)
    temporary.replace(path / "records.jsonl")
    manifest = {
        **metadata,
        "schema": "saebase.dataset.fixed_cohort.v1",
        "dataset": str(cfg.name),
        "requested_count": cfg.get("sample_count", 500),
        "selected_count": len(rows),
        "sampling_seed": cfg.get("sampling_seed", 42),
        "split_policy": "union_all_splits; original labels retained",
        "records_sha256": hashlib.sha256(data).hexdigest(),
        "sample_ids_sha256": hashlib.sha256(
            json.dumps([r["sample_id"] for r in rows]).encode()
        ).hexdigest(),
    }
    write_json(path / "manifest.json", manifest)
    return read_selection(cfg, model_key)


def select_dataset(cfg, *, model_key, model=None, loaded_model_name=None, seed=42):
    import fcntl
    from saebase.registry import get_dataset_plugin

    validate_selection(cfg)
    path = selection_path(cfg, model_key)
    if (path / "manifest.json").is_file():
        return read_selection(cfg, model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (path / "manifest.json").is_file():
            return read_selection(cfg, model_key)
        if cfg.get("prompt_file"):
            from saebase.datasets.sampling import sample_records

            from hydra.utils import to_absolute_path

            source = Path(to_absolute_path(str(cfg.prompt_file))).expanduser().resolve()
            data = source.read_bytes()
            rows = (
                [json.loads(line) for line in data.splitlines() if line.strip()]
                if source.suffix == ".jsonl"
                else json.loads(data)
            )
            for index, row in enumerate(rows):
                row.setdefault("sample_id", f"{cfg.name}:file:{index}")
            chosen = sample_records(
                len(rows),
                rows.__getitem__,
                cfg.get("sample_count", 500),
                cfg.get("sampling_seed", 42),
            )
            if str(cfg.name) == "IOI" and len(chosen) != cfg.get("sample_count", 500):
                raise ValueError(
                    "Synthetic explicit input contains fewer than sample_count eligible records"
                )
            result = SampleSelection(
                chosen,
                {
                    "source": "explicit_file",
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                    "population_count": len(rows),
                },
            )
        else:
            plugin = get_dataset_plugin(str(cfg.name))
            if plugin.select_samples is None:
                raise ValueError(f"Dataset {cfg.name} has no fixed-cohort selector")
            result = plugin.select_samples(
                cfg,
                model_key=model_key,
                model=model,
                loaded_model_name=loaded_model_name,
                seed=cfg.get("sampling_seed", 42),
            )
        return save_selection(cfg, model_key, list(result), result.metadata)
