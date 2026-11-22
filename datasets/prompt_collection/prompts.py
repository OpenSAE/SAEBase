"""Read prepared NLP prompt artifacts from selected dataset and model.

SAEBase's canonical layout is <data_root>/<dataset>/<model>/v1/prompts.jsonl
with a matching manifest.json. Legacy JSON snapshots remain supported.
No implicit downloads, synthetic prompts, or train/test split inventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saebase.datasets.artifacts import prepared_dataset_root, resolve_prepared_path

LEGACY_FILENAMES = {
    "HotpotQA": (),
    "RAVEL": (),
    "ARCChallenge": (),
    "GSM8K": (),
    "HumanEval": (),
    "TruthfulQA": (),
    "HarmBench": (),
    "XSTest": (),
    "CounterFact": ("counterfact_filtered.json", "prompts.json"),
    "GreaterThan": ("greater_than_prompts.json", "prompts.json"),
    "IOI": ("ioi_prompts.json", "prompts.json"),
    "MMLUPro": ("mmlu_pro_prompts.json", "mmlu_pro.json", "prompts.json"),
    "SimpleCopying": ("copying_prompts.json", "prompts.json"),
    "SubjectVerbAgreement": ("sva_prompts.json", "prompts.json"),
    "BBH": ("hf_cache.jsonl", "prompts.json"),
}


def prompt_file(dataset_cfg: Any, *, model_key: str | None = None) -> Path:
    if "sample_count" in dataset_cfg:
        from saebase.datasets.selection import selection_path

        path = selection_path(dataset_cfg, str(model_key or "_shared")) / "records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Selected Dataset cohort is not prepared: {path}")
        return path
    name = str(dataset_cfg.name)
    if name not in LEGACY_FILENAMES:
        raise ValueError(f"Unknown NLP prompt dataset: {name}")
    root = resolve_prepared_path(Path(str(dataset_cfg.get("data_path", "") or "")))
    override = str(dataset_cfg.get("prompt_file", "") or "").strip()
    if override:
        path = Path(override).expanduser()
        path = resolve_prepared_path(path if path.is_absolute() else root / path)
        if not path.is_file():
            raise FileNotFoundError(f"{name} requires prepared prompt file {path}")
        return path
    if model_key:
        candidate = root / str(model_key) / "v1" / "prompts.jsonl"
        if candidate.is_file():
            return candidate
    else:
        matches = sorted(root.glob("*/v1/prompts.jsonl"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Specify model_key when multiple SAEBase prompt sets are present")
    for fname in LEGACY_FILENAMES[name]:
        path = root / fname
        if path.is_file():
            return path
    expected = root / str(model_key or "<model>") / "v1" / "prompts.jsonl"
    raise FileNotFoundError(f"{name} needs prepared prompt records at {expected}")


def read_prompts(dataset_cfg: Any, *, model_key: str | None = None) -> list[dict[str, Any]]:
    if "sample_count" in dataset_cfg:
        from saebase.datasets.selection import read_selection

        return list(read_selection(dataset_cfg, str(model_key or "_shared")))
    path = prompt_file(dataset_cfg, model_key=model_key)
    name = str(dataset_cfg.name)
    if path.suffix == ".jsonl":
        if path.name == "prompts.jsonl" and path.parent.name == "v1":
            # Real SAEBase artifact: verify the manifest identifies its source.
            from saebase.datasets.artifacts import load_prompts

            expected_model = path.parent.parent.name
            if model_key is not None and expected_model != model_key:
                raise ValueError("Prompt artifact model does not match selected model")
            rows = load_prompts(
                name, expected_model, data_root=path.parents[3], require_manifest=True
            )
        else:
            with path.open(encoding="utf-8") as source:
                rows = [json.loads(line) for line in source if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("prompts", rows.get("rows", rows.get("samples")))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected nonempty prompt list in {path}")
    valid = []
    for index, item in enumerate(rows):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index}: expected prompt object or string")
        prompt = item.get("prompt", item.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{index}: missing prompt text")
        valid.append({**item, "prompt": prompt.strip()})
    return valid


def read_first_prompt(dataset_cfg: Any, *, model_key: str | None = None) -> str:
    return read_prompts(dataset_cfg, model_key=model_key)[0]["prompt"]


def read_measurement_samples(dataset_cfg: Any) -> list[dict[str, Any]]:
    """Read explicitly tokenized measurement records; preserve supplied split and token IDs."""
    from hydra.utils import to_absolute_path
    from omegaconf import OmegaConf

    samples = dataset_cfg.get("samples")
    source = dataset_cfg.get("sample_file")
    if samples is not None and source:
        raise ValueError("Select dataset.samples or dataset.sample_file, not both")
    if source:
        path = resolve_prepared_path(Path(to_absolute_path(str(source))))
        if path.suffix == ".jsonl":
            samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            samples = json.loads(path.read_text())
    elif OmegaConf.is_config(samples):
        samples = OmegaConf.to_container(samples, resolve=True)
    if (
        not isinstance(samples, list)
        or not samples
        or any(not isinstance(row, dict) for row in samples)
    ):
        raise ValueError(
            "Provide nonempty dataset.samples or dataset.sample_file measurement records"
        )
    return samples


def prepare_selected_dataset(
    cfg: Any, *, n_generate: int = 500, min_kept: int = 1
) -> dict[str, Any]:
    """Prepare missing real records through their owner, staging before directory commit."""
    import fcntl
    import os
    import tempfile

    from models.loading_policy import resolve_config_devices

    resolve_config_devices(cfg)

    from saebase.registry import get_dataset_plugin, get_model_plugin
    from saebase.datasets.artifacts import artifact_dir

    if "sample_count" in cfg.dataset:
        from saebase.datasets.selection import select_dataset, selection_path

        model_key = str(cfg.model.name)
        handle = None
        if (
            str(cfg.dataset.name) == "IOI"
            and not (selection_path(cfg.dataset, model_key) / "manifest.json").exists()
        ):
            from saebase.runtime.sae_operations import _ACTIVE_PAIR

            pair = _ACTIVE_PAIR.get()
            handle = pair.model if pair is not None else get_model_plugin(model_key).load_model(cfg)
        selected = select_dataset(
            cfg.dataset,
            model_key=model_key,
            model=getattr(handle, "model", handle),
            loaded_model_name=str(getattr(handle, "loaded_name", model_key)),
        )
        return {
            "status": "READY",
            "path": str(selection_path(cfg.dataset, model_key) / "records.jsonl"),
            "records": selected.metadata["selected_count"],
            "manifest": selected.metadata,
        }
    if type(n_generate) is not int or type(min_kept) is not int or not 0 < min_kept <= n_generate:
        raise ValueError("Require 0 < dataset_min_kept <= dataset_n_generate")
    model_key = str(cfg.model.name)
    try:
        path = prompt_file(cfg.dataset, model_key=model_key)
    except FileNotFoundError:
        path = None
    if path is not None:
        rows = read_prompts(cfg.dataset, model_key=model_key)
        return {"status": "READY", "source": "existing", "path": str(path), "records": len(rows)}
    if cfg.dataset.get("prompt_file"):
        raise FileNotFoundError(
            "Explicit dataset.prompt_file is missing; refusing to replace its source"
        )
    plugin = get_dataset_plugin(str(cfg.dataset.name))
    if plugin.prepare is None:
        raise ValueError(f"Dataset {cfg.dataset.name} has no preparation method")
    root = Path(str(cfg.paths.data_dir)).expanduser().resolve()
    dataset_root = (
        Path(str(cfg.dataset.get("data_path") or prepared_dataset_root(root, str(cfg.dataset.name))))
        .expanduser()
        .absolute()
    )
    dataset_root = resolve_prepared_path(dataset_root)
    if dataset_root != prepared_dataset_root(dataset_root.parent, str(cfg.dataset.name)):
        raise ValueError("Prepared dataset.data_path must end with the selected Dataset name")
    destination = artifact_dir(str(cfg.dataset.name), model_key, data_root=dataset_root.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent or destination.is_symlink():
        raise ValueError("Unsafe dataset preparation location")
    lock = destination.parent / ".prepare.lock"
    if lock.is_symlink():
        raise ValueError("Unsafe dataset preparation lock")
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Dataset preparation already has an active writer") from exc
        if destination.exists():
            raise FileExistsError(
                "Existing dataset artifacts are incomplete; preserved without overwrite"
            )
        handle = get_model_plugin(model_key).load_model(cfg)
        model = getattr(handle, "model", handle)
        loaded = str(getattr(handle, "loaded_name", model_key))
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as stage:
            prepared = plugin.prepare(
                model_key,
                model,
                loaded_model_name=loaded,
                model_name=str(
                    cfg.model.get("hf_id") or cfg.model.get("pretrained_name") or loaded
                ),
                n_generate=n_generate,
                min_kept=min_kept,
                seed=int(cfg.seed),
                data_root=stage,
            )
            expected = artifact_dir(str(cfg.dataset.name), model_key, data_root=stage)
            if Path(prepared).resolve() != expected.resolve():
                raise ValueError("Dataset preparer returned a different input identity")
            source_cfg = {
                "name": str(cfg.dataset.name),
                "data_path": str(expected.parent.parent),
                "prompt_file": str(expected / "prompts.jsonl"),
            }
            from omegaconf import OmegaConf

            rows = read_prompts(OmegaConf.create(source_cfg), model_key=model_key)
            manifest = json.loads((expected / "manifest.json").read_text())
            if manifest["model_key"] != model_key or manifest["behavior"] != cfg.dataset.name:
                raise ValueError("Prepared Dataset manifest identity mismatch")
            os.rename(expected, destination)
        return {
            "status": "READY",
            "source": "prepared",
            "path": str(destination / "prompts.jsonl"),
            "records": len(rows),
            "manifest": manifest,
        }
