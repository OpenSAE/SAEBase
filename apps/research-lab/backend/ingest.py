"""CPU-only ingestion of locally exported datasets and genuine SAE observations.

This module never downloads weights, guesses licensing rights, or invents results.
Run from apps/research-lab with: python -m backend.ingest --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .core import canonical_json, data_root, dataset_catalog, load_sites, has_complete_identity


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing nonempty " + field)
    return value.strip()


def _choice_labels(choices: Any) -> tuple[list[str], list[str]]:
    if isinstance(choices, dict):
        labels, texts = choices.get("label"), choices.get("text")
    else:
        labels = [str(i) for i in range(len(choices))] if isinstance(choices, list) else None
        texts = choices
    if (
        not isinstance(labels, list)
        or not isinstance(texts, list)
        or len(labels) != len(texts)
        or not labels
    ):
        raise ValueError("Invalid candidate choices")
    return [str(x) for x in labels], [_text(x, "choice") for x in texts]


def normalize(source: str, row: dict[str, Any]) -> dict[str, Any]:
    """Render a base *question/prefix*, never append the withheld gold answer."""
    raw_id = str(row.get("id") or row.get("idx") or row.get("qID") or row.get("task_id") or "")
    label = None
    candidates = None
    candidate_texts = None
    reference_answers = None
    if source == "wikitext103":
        prompt = _text(row.get("text"), "text")
        group = _text(row.get("article_id") or row.get("document_id"), "article_id")
        raw_id = raw_id or hashlib.sha256(prompt.encode()).hexdigest()[:20]
    elif source == "tinystories":
        prompt = _text(row.get("text"), "text")
        group = raw_id or hashlib.sha256(prompt.encode()).hexdigest()[:20]
        raw_id = raw_id or group
    elif source == "squad":
        context = _text(row.get("context"), "context")
        question = _text(row.get("question"), "question")
        prompt = "Context:\n" + context + "\nQuestion: " + question + "\nAnswer:"
        group = _text(row.get("title") or row.get("article_id"), "article title")
        raw_id = _text(raw_id, "question id")
        answers = row.get("answers")
        if isinstance(answers, dict) and isinstance(answers.get("text"), list):
            reference_answers = [_text(text, "answer") for text in answers["text"]]
    elif source == "boolq":
        context = _text(row.get("passage"), "passage")
        question = _text(row.get("question"), "question")
        prompt = "Passage:\n" + context + "\nQuestion: " + question + "\nAnswer yes or no:"
        group = hashlib.sha256(context.encode()).hexdigest()[:20]
        answer = row.get("answer")
        if type(answer) is not bool:
            raise ValueError("BoolQ must contain boolean answer")
        label = "yes" if answer else "no"
        candidates = ["yes", "no"]
        candidate_texts = ["yes", "no"]
        raw_id = raw_id or hashlib.sha256((context + question).encode()).hexdigest()[:20]
    elif source == "arc_challenge":
        question = _text(row.get("question"), "question")
        labels, texts = _choice_labels(row.get("choices"))
        prompt = (
            question + "\n" + "\n".join(a + ". " + b for a, b in zip(labels, texts)) + "\nAnswer:"
        )
        group = raw_id or hashlib.sha256(question.encode()).hexdigest()[:20]
        label = str(row.get("answerKey", ""))
        candidates = labels
        candidate_texts = texts
        if label not in candidates:
            raise ValueError("ARC answer does not match choices")
        raw_id = raw_id or group
    elif source == "hellaswag":
        context = _text(
            row.get("ctx") or (str(row.get("ctx_a", "")) + " " + str(row.get("ctx_b", ""))).strip(),
            "ctx",
        )
        endings = row.get("endings")
        if not isinstance(endings, list) or len(endings) != 4:
            raise ValueError("HellaSwag requires 4 endings")
        candidates = ["A", "B", "C", "D"]
        candidate_texts = [_text(ending, "ending") for ending in endings]
        prompt = (
            context
            + "\n"
            + "\n".join(ch + ". " + _text(t, "ending") for ch, t in zip(candidates, endings))
            + "\nAnswer:"
        )
        raw_id = raw_id or hashlib.sha256(context.encode()).hexdigest()[:20]
        group = raw_id
        label_index = str(row.get("label", ""))
        if label_index not in ("0", "1", "2", "3"):
            raise ValueError("HellaSwag validation label required")
        label = candidates[int(label_index)]
    elif source == "winogrande":
        sentence = _text(row.get("sentence"), "sentence")
        if sentence.count("_") != 1:
            raise ValueError("WinoGrande requires exactly one blank")
        one, two = _text(row.get("option1"), "option1"), _text(row.get("option2"), "option2")
        prompt = sentence + "\n1. " + one + "\n2. " + two + "\nAnswer:"
        raw_id = raw_id or hashlib.sha256(sentence.encode()).hexdigest()[:20]
        group = hashlib.sha256(sentence.encode()).hexdigest()[:20]
        label = str(row.get("answer", ""))
        if label not in ("1", "2"):
            raise ValueError("WinoGrande requires labeled validation set")
        candidates = ["1", "2"]
        candidate_texts = [one, two]
    elif source == "gsm8k":
        question = _text(row.get("question"), "question")
        prompt = "Question: " + question + "\nAnswer:"
        raw_id = raw_id or hashlib.sha256(question.encode()).hexdigest()[:20]
        group = raw_id
        official = row.get("answer")
        if isinstance(official, str) and "####" in official:
            reference_answers = [_text(official.rsplit("####", 1)[-1], "final answer")]
    elif source == "humaneval":
        prompt = _text(row.get("prompt"), "prompt")
        raw_id = _text(row.get("task_id"), "task_id")
        group = raw_id
    elif source == "truthfulqa":
        question = _text(row.get("question"), "question")
        mc = row.get("mc1_targets")
        if not isinstance(mc, dict) or not isinstance(mc.get("choices"), list):
            raise ValueError("TruthfulQA multiple_choice mc1_targets missing")
        labels = [chr(65 + i) for i in range(len(mc["choices"]))]
        if not (2 <= len(labels) <= 26):
            raise ValueError("Unsupported TruthfulQA choice count")
        prompt = (
            question
            + "\n"
            + "\n".join(ch + ". " + _text(s, "choice") for ch, s in zip(labels, mc["choices"]))
            + "\nAnswer:"
        )
        scores = mc.get("labels")
        if (
            not isinstance(scores, list)
            or len(scores) != len(labels)
            or sum(x == 1 for x in scores) != 1
        ):
            raise ValueError("TruthfulQA MC1 requires exactly one true choice")
        label = labels[scores.index(1)]
        candidates = labels
        candidate_texts = [_text(text, "choice") for text in mc["choices"]]
        raw_id = raw_id or hashlib.sha256(question.encode()).hexdigest()[:20]
        group = raw_id
    else:
        raise ValueError("Unknown dataset " + source)
    if not raw_id or not group or not prompt:
        raise ValueError("Invalid source identity")
    if len(prompt) > 30000:
        raise ValueError("Unexpected input length; audit before truncating")
    return {
        "source_row_id": raw_id,
        "source_group_id": str(group),
        "prompt": prompt,
        "task_label": label,
        "candidate_labels": candidates,
        "candidate_texts": candidate_texts,
        "reference_answers": reference_answers,
        "render_protocol": (
            "opensae.render.choice_label.v1"
            if candidates is not None
            else "opensae.render.question_answer.v1"
            if source in ("squad", "gsm8k")
            else "opensae.render.code_prefix.v1"
            if source == "humaneval"
            else "opensae.render.continuation.v1"
        ),
        "evaluation_warning": (
            "HellaSwag multiple-choice label rendering is not the official continuation-likelihood protocol"
            if source == "hellaswag"
            else None
        ),
    }


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "\x00" + "\x00".join(parts)).encode("utf-8")).hexdigest()


def select_records(
    source: str,
    rows: Iterable[dict[str, Any]],
    *,
    seed: int = 20261018,
    quota: int = 50,
) -> list[dict[str, Any]]:
    """Reproducibly sample distinct source groups before model outputs are seen."""
    if quota != 50:
        raise ValueError("Public v1 requires exactly 50 per source")
    picked: dict[str, dict[str, Any]] = {}
    prompts: set[str] = set()
    for row in rows:
        try:
            normal = normalize(source, row)
        except (ValueError, TypeError, KeyError):
            continue
        group = normal["source_group_id"]
        # Keep exactly one deterministic row within each original group.
        rank = _rank(seed, source, group, normal["source_row_id"])
        if group not in picked or rank < picked[group]["_rank"]:
            normal["_rank"] = rank
            picked[group] = normal
    eligible = []
    for group, value in sorted(
        picked.items(), key=lambda item: (_rank(seed, source, item[0]), item[0])
    ):
        fingerprint = hashlib.sha256(value["prompt"].encode()).hexdigest()
        if fingerprint in prompts:
            continue
        prompts.add(fingerprint)
        value["_group_rank"] = _rank(seed, source, group)
        eligible.append(value)
    if len(eligible) < quota:
        raise ValueError(f"{source}: {len(eligible)} distinct eligible groups, need {quota}")
    eligible.sort(key=lambda r: r["_group_rank"])
    # BoolQ is balanced by its source gold class; no model-generated outcome is used.
    if source == "boolq":
        yes = [r for r in eligible if r["task_label"] == "yes"][:25]
        no = [r for r in eligible if r["task_label"] == "no"][:25]
        if len(yes) != 25 or len(no) != 25:
            raise ValueError("BoolQ requires 25 true and 25 false")
        eligible = sorted(yes + no, key=lambda r: r["_group_rank"])
    elif source == "hellaswag":
        # Select from the frozen gold labels, never model correctness.
        # Fifty samples cannot divide equally into four classes, so use
        # deterministic counts 13/13/12/12 for A/B/C/D.
        quotas = {"A": 13, "B": 13, "C": 12, "D": 12}
        selected = []
        for label, count in quotas.items():
            group = [row for row in eligible if row["task_label"] == label][:count]
            if len(group) != count:
                raise ValueError("HellaSwag cannot satisfy fixed gold-label quotas")
            selected.extend(group)
        eligible = sorted(selected, key=lambda row: row["_group_rank"])
    else:
        eligible = eligible[:quota]
    for index, record in enumerate(eligible):
        record.pop("_rank", None)
        record.pop("_group_rank", None)
        record["dataset_id"] = source
        record["prompt_group_id"] = source + ":" + record["source_group_id"]
        record["sample_id"] = source + ":" + record["source_row_id"]
        record["usage_split"] = "discovery" if index < 20 else "evaluation"
    return eligible


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze_dataset(
    source: str,
    rows: Iterable[dict[str, Any]],
    *,
    source_revision: str,
    output: Path,
    license_reviewed: bool = False,
    human_reviewed: bool = False,
    seed: int = 20261018,
) -> dict[str, Any]:
    sources = {d["id"]: d for d in dataset_catalog()["sources"]}
    if source not in sources:
        raise ValueError("Unknown source")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(ch not in "0123456789abcdef" for ch in source_revision)
    ):
        raise ValueError("Pin exact 40-character lowercase upstream SHA (not main/latest)")
    records = select_records(source, rows, seed=seed)
    if (
        len({r["prompt_group_id"] for r in records}) != 50
        or len({r["sample_id"] for r in records}) != 50
    ):
        raise ValueError("Source-group or sample identity collision")
    for row in records:
        row["source_revision"] = source_revision
        row["source_repo"] = sources[source]["repo"]
        row["source_split"] = sources[source]["split"]
    data = b"".join(canonical_json(r) for r in records)
    meta = {
        "schema": "opensae.dataset_release.v1",
        "dataset_id": source,
        "source_repo": sources[source]["repo"],
        "source_config": sources[source]["config"],
        "source_split": sources[source]["split"],
        "source_revision": source_revision,
        "seed": seed,
        "sample_count": len(records),
        "discovery": 20,
        "evaluation": 30,
        "records_sha256": hashlib.sha256(data).hexdigest(),
        "license_reviewed": bool(license_reviewed),
        "human_reviewed": bool(human_reviewed),
        "interpretation": "curated subset; neither unseen training data nor large-scale benchmark",
    }
    if (output / "manifest.json").exists():
        raise FileExistsError("Immutable dataset release already exists")
    _atomic_bytes(output / "records.jsonl", data)
    _atomic_bytes(output / "manifest.json", canonical_json(meta))
    return meta


def build_feature_index(
    observations: Iterable[dict[str, Any]], *, output: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build compact browsing index from REAL activation records only.

    Each observation must bind to an exact verified site & model/SAE weights.
    Text is shown only if an explicit rights review permits publication.
    """
    sites = {s["site_id"]: s for s in load_sites(repo_root)}
    cards: dict[str, dict[str, Any]] = {}
    for row in observations:
        site_id = row.get("site_id")
        site = sites.get(site_id)
        if (
            site is None
            or not has_complete_identity(site)
            or row.get("model_sha256") != site["model_sha256"]
            or row.get("sae_sha256") != site["sae_sha256"]
        ):
            raise ValueError("Observation does not match a verified model/SAE/site identity")
        if row.get("source_type") != "real_checkpoint":
            raise ValueError("Synthetic or unspecified observations cannot be published as real")
        feature_id = row.get("feature_id")
        position = row.get("token_position")
        if (
            type(feature_id) is not int
            or feature_id < 0
            or type(position) is not int
            or position < 0
        ):
            raise ValueError("Invalid feature ID or token position")
        strength = row.get("activation")
        if not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise ValueError("Nonfinite activation")
        group = _text(row.get("prompt_group_id"), "prompt_group_id")
        key = site_id + ":F" + str(feature_id)
        record = cards.setdefault(
            key,
            {
                "feature_key": key,
                "site_id": site_id,
                "model": site["model"],
                "sae": site["sae"],
                "layer": site["layer"],
                "feature_id": feature_id,
                "observations": 0,
                "max_activation": float("-inf"),
                "groups": set(),
                "examples": [],
                "source_type": "real_checkpoint",
                "label": None,
                "label_status": "unlabeled",
            },
        )
        record["observations"] += 1
        record["groups"].add(group)
        record["max_activation"] = max(record["max_activation"], float(strength))
        if bool(row.get("public_text_reviewed")):
            example = {
                "prompt_group_id": group,
                "sample_id": str(row.get("sample_id", group)),
                "token_position": position,
                "token_text": str(row.get("token_text", ""))[:128],
                "text_excerpt": str(row.get("text_excerpt", ""))[:700],
                "activation": float(strength),
            }
            record["examples"].append(example)
            record["examples"].sort(key=lambda r: -r["activation"])
            del record["examples"][8:]
    for item in cards.values():
        item["unique_prompt_groups"] = len(item.pop("groups"))
        if item["max_activation"] == float("-inf"):
            item["max_activation"] = None
    data = canonical_json(
        sorted(cards.values(), key=lambda v: (v["model"], v["layer"], v["feature_id"]))
    )
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("Feature index exceeds viewer budget; shard it")
    target = output / "index.json"
    if (output / "manifest.json").exists():
        raise FileExistsError("Feature index is immutable")
    _atomic_bytes(target, data)
    manifest = {
        "schema": "opensae.feature_index.v1",
        "files": {"index.json": hashlib.sha256(data).hexdigest()},
        "feature_count": len(cards),
    }
    _atomic_bytes(output / "manifest.json", canonical_json(manifest))
    return manifest


def _raw_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected JSON object at line " + str(number))
            yield row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ds = sub.add_parser("dataset", help="Freeze 50 rows from a locally exported dataset")
    ds.add_argument(
        "--source", required=True, choices=[s["id"] for s in dataset_catalog()["sources"]]
    )
    ds.add_argument("--input", type=Path, required=True)
    ds.add_argument("--source-revision", required=True)
    ds.add_argument("--output", type=Path, default=None)
    ds.add_argument("--license-reviewed", action="store_true")
    ds.add_argument("--human-reviewed", action="store_true")
    fi = sub.add_parser("features", help="Build a real-only feature index from GPU output")
    fi.add_argument("--input", type=Path, required=True)
    fi.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "dataset":
        output = args.output or (data_root() / "datasets" / args.source)
        report = freeze_dataset(
            args.source,
            _raw_jsonl(args.input),
            source_revision=args.source_revision,
            output=output,
            license_reviewed=args.license_reviewed,
            human_reviewed=args.human_reviewed,
        )
    else:
        report = build_feature_index(
            _raw_jsonl(args.input), output=args.output or data_root() / "features"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
