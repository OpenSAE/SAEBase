"""Real public Dataset acquisition and bounded prompt preparation.

Owner: Dataset. Reuses HF caches and the canonical prompt store; never evaluates
model accuracy, executes code, or manufactures benchmark scores. Source records
and upstream splits are retained alongside deterministic bounded observations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from saebase.datasets.artifacts import default_data_root
from saebase.datasets.artifacts import assert_available

# A concrete split and configuration, not a second component registry.
HF_SOURCES = {
    "ARCChallenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "GSM8K": ("openai/gsm8k", "main", "test"),
    "HumanEval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "TruthfulQA": ("truthfulqa/truthful_qa", "multiple_choice", "validation"),
}


def _parquet(repo: str, filename: str, revision: str) -> list[dict]:
    import pyarrow.parquet as pq

    file = hf_hub_download(repo, filename, revision=revision, repo_type="dataset")
    return pq.read_table(file).to_pylist()


def _acquire(
    name: str, limit: int, *, revision_override: str | None = None
) -> tuple[list[dict], dict]:
    api = HfApi()
    if name in HF_SOURCES:
        repo, config, split = HF_SOURCES[name]
        info = api.dataset_info(repo, revision=revision_override)
        paths = sorted(
            s.rfilename
            for s in info.siblings
            if s.rfilename.startswith(config + "/")
            and s.rfilename.endswith(".parquet")
            and Path(s.rfilename).name.startswith(split)
        )
        if not paths:
            raise ValueError(f"No {config}/{split} parquet source in {repo}@{info.sha}")
        rows = []
        for path in paths:
            rows.extend(_parquet(repo, path, info.sha))
            if len(rows) >= limit:
                break
        files = paths
        return rows, {
            "repo": repo,
            "revision": info.sha,
            "config": config,
            "upstream_split": split,
            "files": files,
        }
    if name == "XSTest":
        repo = "Paul/XSTest"
        revision = api.dataset_info(repo, revision=revision_override).sha
        file = hf_hub_download(repo, "xstest_prompts.csv", repo_type="dataset", revision=revision)
        return list(csv.DictReader(Path(file).open())), {
            "repo": repo,
            "revision": revision,
            "upstream_split": "unspecified",
            "file": "xstest_prompts.csv",
        }
    if name == "HarmBench":
        import requests

        repo = "centerforaisafety/HarmBench"
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/main", timeout=60)
        response.raise_for_status()
        revision = response.json()["sha"]
        file = "data/behavior_datasets/harmbench_behaviors_text_test.csv"
        response = requests.get(
            f"https://raw.githubusercontent.com/{repo}/{revision}/{file}", timeout=60
        )
        response.raise_for_status()
        return list(csv.DictReader(io.StringIO(response.text))), {
            "repo": repo,
            "revision": revision,
            "file": file,
            "upstream_split": "test",
            "raw_sha256": hashlib.sha256(response.content).hexdigest(),
        }
    if name == "RAVEL":
        repo = "hij/ravel"
        revision = api.dataset_info(repo, revision=revision_override).sha
        entities = _parquet(repo, "city_entity/test-00000-of-00001.parquet", revision)
        templates = _parquet(repo, "city_prompt/test-00000-of-00001.parquet", revision)
        rows = []
        for entity in entities:
            for template in templates:
                if template.get("Source") != "RAVEL" or template["Template"].count("%s") != 1:
                    continue
                attribute = template["Attribute"]
                if attribute not in entity:
                    continue
                rows.append(
                    {
                        "id": f"{entity['ID']}:{len(rows)}",
                        "prompt": template["Template"] % entity["City"],
                        "entity": entity["City"],
                        "attribute": attribute,
                        "reference_answer": entity[attribute],
                        "entity_row": entity,
                        "template_row": template,
                    }
                )
                if len(rows) >= limit:
                    return rows, {
                        "repo": repo,
                        "revision": revision,
                        "upstream_split": "test",
                        "scope": "city entity test x city prompt test; no interchange intervention scoring",
                    }
        raise ValueError("RAVEL produced too few valid entity/template records")
    raise ValueError(f"Unknown public Dataset {name}")


def source_records(
    name: str, *, limit: int = 500, source_root: Path | None = None
) -> tuple[list[dict], dict]:
    """Cache selected raw records with immutable upstream revision and content hash."""
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("Source limit must be 1..10000")
    directory = (source_root or default_data_root() / "sources") / name / f"prefix_{limit}"
    source, manifest = directory / "records.json", directory / "source.json"
    if source.is_file() and manifest.is_file():
        data = source.read_bytes()
        meta = json.loads(manifest.read_text())
        if hashlib.sha256(data).hexdigest() != meta["records_sha256"]:
            raise ValueError("Source cache checksum mismatch")
        return json.loads(data), meta
    rows, meta = _acquire(name, max(limit * 4, 128))
    chosen, seen = [], set()
    for index, row in enumerate(rows):
        prompt = normalize(name, row)
        if not prompt or len(prompt) > 4000 or prompt in seen:
            continue
        seen.add(prompt)
        chosen.append({"source_index": index, "raw": row, "prompt": prompt})
        if len(chosen) == limit:
            break
    if not chosen:
        raise ValueError(f"No usable real prompts from {name}")
    data = json.dumps(chosen, ensure_ascii=False, sort_keys=True).encode()
    meta.update(
        schema="saebase.public_source.v1",
        selection="first unique nonempty prompts up to4000 characters, no truncation",
        requested=limit,
        retained=len(chosen),
        full_dataset_download=False,
        records_sha256=hashlib.sha256(data).hexdigest(),
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Unique temporary files and atomic replacement: interrupted writes are never a valid cache.
    for path, content in [(source, data), (manifest, json.dumps(meta, indent=2).encode())]:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return chosen, meta


def normalize(name: str, row: dict) -> str:
    if name == "RAVEL":
        return row["prompt"].strip()
    if name == "ARCChallenge":
        c = row["choices"]
        return (
            row["question"]
            + "\n"
            + "\n".join(
                f"{label}. {text}" for label, text in zip(c["label"], c["text"], strict=True)
            )
            + "\nAnswer:"
        )
    if name == "GSM8K":
        return row["question"].strip() + "\nAnswer:"
    if name == "HumanEval":
        return row["prompt"].strip()
    if name == "TruthfulQA":
        choices = row["mc1_targets"]["choices"]
        return (
            row["question"]
            + "\n"
            + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(choices))
            + "\nAnswer:"
        )
    if name == "HarmBench":
        if row.get("FunctionalCategory", "standard") not in {"standard", "copyright"}:
            return ""
        return row["Behavior"].strip()
    if name == "XSTest":
        return row["prompt"].strip()
    raise ValueError(f"Unknown Dataset normalizer {name}")


def prepare_public_dataset(
    name: str,
    model_key: str,
    model: Any,
    *,
    loaded_model_name: str,
    model_name: str,
    n_generate: int = 500,
    seed: int = 42,
    min_kept: int = 1,
    data_root: str | Path | None = None,
) -> Path:
    """Prepare the Dataset-owned fixed cohort, independent of candidate flags.

    n_generate/min_kept remain accepted for source compatibility. Natural input
    counts come from the Dataset configuration, never from synthetic budgets.
    """
    from importlib.resources import files
    from omegaconf import OmegaConf
    from saebase.datasets.selection import select_dataset, selection_path

    assert_available(name, model_key)
    root = Path(data_root) if data_root is not None else default_data_root()
    definition = None
    for path in (files("conf") / "dataset").iterdir():
        if path.name.endswith(".yaml"):
            candidate = OmegaConf.load(path).get("dataset")
            if candidate is not None and candidate.get("name") == name:
                definition = candidate
                break
    if definition is None:
        raise ValueError(f"Missing Dataset configuration: {name}")
    cfg = OmegaConf.merge(definition, {"data_path": str(root / name), "sampling_seed": seed})
    select_dataset(cfg, model_key=model_key)
    return selection_path(cfg, model_key)


def _render_record(name, raw, index, split, configuration=None):
    if name == "CounterFact":
        from saebase.datasets.CounterFact.CounterFact import normalize_counterfact_schema

        data = normalize_counterfact_schema([raw])
        prompt = data[0]["prompt"] if data else ""
    elif name == "MMLUPro":
        prompt = (
            raw["question"]
            + "\n"
            + "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(raw["options"]))
            + "\nAnswer:"
        )
    elif name == "HotpotQA":
        context = raw["context"]
        titles = context["title"]
        sentences = context["sentences"]
        prompt = (
            "\n".join(
                title + ": " + " ".join(parts)
                for title, parts in zip(titles, sentences, strict=True)
            )
            + "\nQuestion: "
            + raw["question"]
            + "\nAnswer:"
        )
    elif name == "BBH":
        prompt = raw["input"].strip()
    elif name == "HarmBench" and raw.get("FunctionalCategory") == "contextual":
        prompt = str(raw.get("ContextString", "")).strip() + "\n\n" + raw["Behavior"].strip()
    else:
        prompt = normalize(name, raw)
    return {
        "sample_id": f"{name}:{configuration or 'default'}:{split}:{index}",
        "prompt": prompt,
        "split": split,
        "source_index": index,
        "source_config": configuration,
        "source_record": raw,
    }


def select_public_samples(name, cfg, *, model_key, **kwargs):
    """Uniform fixed cohort from every split; raw split labels are retained only."""
    from saebase.datasets.selection import SampleSelection
    from saebase.datasets.sampling import sample_records, union_index

    count = cfg.get("sample_count", 500)
    seed = cfg.get("sampling_seed", 42)
    metadata = {
        "dataset": name,
        "sampling": "uniform_without_replacement_across_all_splits",
        "filter_policy": "no_model_accuracy_filter",
    }
    if name == "RAVEL":
        if cfg.get("source_config", "city") != "city":
            raise ValueError("RAVEL currently supports the city source configuration")
        import bisect

        repo = "hij/ravel"
        revision = HfApi().dataset_info(repo, revision=cfg.get("source_revision")).sha
        entities = []
        templates = []
        for split in ("train", "val", "test"):
            entities.extend(
                {**r, "_split": split}
                for r in _parquet(repo, f"city_entity/{split}-00000-of-00001.parquet", revision)
            )
            templates.extend(
                {**r, "_split": split}
                for r in _parquet(repo, f"city_prompt/{split}-00000-of-00001.parquet", revision)
                if r["Source"] == "RAVEL" and r["Template"].count("%s") == 1
            )
        choices = [[i for i, t in enumerate(templates) if t["Attribute"] in e] for e in entities]
        bounds = []
        total = 0
        for indices in choices:
            total += len(indices)
            bounds.append(total)

        def get(index):
            ei = bisect.bisect_right(bounds, index)
            local = index - (bounds[ei - 1] if ei else 0)
            ti = choices[ei][local]
            e, t = entities[ei], templates[ti]
            return {
                "sample_id": f"RAVEL:{e['ID']}:{ti}",
                "prompt": (t["Template"] % e["City"]).strip(),
                "split": e["_split"],
                "template_split": t["_split"],
                "reference_answer": e[t["Attribute"]],
                "entity": e,
                "template": t,
            }

        rows = sample_records(total, get, count, seed)
        metadata.update(
            repo=repo,
            revision=revision,
            population_count=total,
            entities=len(entities),
            templates=len(templates),
        )
    elif name in {"HarmBench", "XSTest"}:
        if name == "XSTest":
            raw, source = _acquire(name, 0, revision_override=cfg.get("source_revision"))
            parts = {"unspecified": raw}
            metadata.update(source)
        else:
            import requests

            repo = "centerforaisafety/HarmBench"
            response = requests.get(
                f"https://api.github.com/repos/{repo}/commits/{cfg.get('source_revision') or 'main'}",
                timeout=60,
            )
            response.raise_for_status()
            revision = response.json()["sha"]
            parts = {}
            for split, suffix in [("validation", "val"), ("test", "test")]:
                response = requests.get(
                    f"https://raw.githubusercontent.com/{repo}/{revision}/data/behavior_datasets/harmbench_behaviors_text_{suffix}.csv",
                    timeout=60,
                )
                response.raise_for_status()
                parts[split] = list(csv.DictReader(io.StringIO(response.text)))
            metadata.update(repo=repo, revision=revision)
        total, get_raw = union_index(parts)

        def get(index):
            split, local, raw = get_raw(index)
            return _render_record(name, raw, local, split)

        rows = sample_records(total, get, count, seed)
        metadata["population_count"] = total
    else:
        from models.hf_compat import import_hf_datasets

        hf = import_hf_datasets()
        sources = {
            **HF_SOURCES,
            "CounterFact": ("azhx/counterfact", "default", "all"),
            "MMLUPro": ("TIGER-Lab/MMLU-Pro", "default", "all"),
            "HotpotQA": ("hotpotqa/hotpot_qa", "distractor", "all"),
            "BBH": ("lukaemon/bbh", None, "all"),
        }
        repo, configuration, _ = sources[name]
        configuration = cfg.get("source_config") or configuration
        revision = HfApi().dataset_info(repo, revision=cfg.get("source_revision")).sha
        configurations = (
            hf.get_dataset_config_names(repo, revision=revision)
            if name == "BBH" and configuration is None
            else [configuration]
        )
        parts = {}
        for configuration in sorted(configurations):
            datasets = hf.load_dataset(repo, name=configuration, revision=revision)
            for split, data in datasets.items():
                parts[(configuration, split)] = data
        total, get_raw = union_index(parts)

        def get(index):
            (configuration, split), local, raw = get_raw(index)
            return _render_record(name, raw, local, split, configuration)

        rows = sample_records(total, get, count, seed)
        metadata.update(
            repo=repo,
            revision=revision,
            population_count=total,
            source_configs=configurations,
            source_split_sizes={f"{c}/{s}": len(v) for (c, s), v in parts.items()},
        )
    return SampleSelection(rows, metadata)
