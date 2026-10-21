#!/usr/bin/env python3
"""Export recorded Feature Effects, controlled-input, and RSS cases.

Reads an existing research-lab run directory. Does not load models, allocate
dictionaries, or write back into the run. Public JSON contains measured
numbers, opaque sample ids, and the project-authored 64-state prompts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RUN_ID = "7fb2be8f29a1-20261019T035727Z"
EFFECTS_SCHEMA = "opensae.lab.effects.v1"
CONTEXT_SCHEMA = "opensae.lab.context64.v1"
RSS_SCHEMA = "opensae.lab.rss.v1"
MANIFEST_SCHEMA = "opensae.lab.interactive_manifest.v1"
LAB_SCHEMA = "opensae.lab.evidence.v1"
EFFECTS_PROTOCOL = "gpu04.paired_operator_logit_l2.v1"
CONTEXT_PROTOCOL = "opensae.controlled_six_factor_inputs.v1"
RSS_PROTOCOL = "rss.position_recomputed.v2"
EFFECTS_URL = "https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation/blob/main/data/research_lab/gpu/intervention_pairs.csv"
CONTEXT_URL = "https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation/blob/main/data/research_lab/controlled/controlled_inputs.csv"
RSS_URL = "https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation/blob/main/data/research_lab/gpu/rss_group_settings.csv"
OPERATOR_ORDER = ("ablate", "scale", "add", "clamp", "coalition")
CONTEXT_SITE = "pythia410m:pythia410m_mlp_topk:L12"
CODES_NAME = "pythia410m_pythia410m_mlp_topk_L12_codes.jsonl"
SETTING_ORDER = ("local_format", "instruction_wrapper", "context_wrapper")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_json(path: Path, payload: object) -> str:
    text = dump_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_site(site_id: str) -> tuple[str, str, int]:
    model, sae, layer = site_id.split(":")
    if not layer.startswith("L") or not layer[1:].isdigit():
        raise ValueError(f"Unrecognized site id: {site_id}")
    return model, sae, int(layer[1:])


def load_registry(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    deep = payload.get("deep")
    if not isinstance(deep, list) or not deep:
        raise SystemExit(f"Site registry {path} has no deep site list")
    return [str(item) for item in deep]


def audit(bucket: list[dict], reason: str, **fields: object) -> None:
    bucket.append({"reason": reason, **fields})


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def build_effects(run_root: Path, deep_sites: list[str], rejected: list[dict]) -> tuple[dict, dict]:
    path = run_root / "GPU-04" / "interventions" / "pairs.json"
    if not path.is_file():
        raise SystemExit(f"Missing effects source: {path}")
    source_sha = sha256_file(path)
    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        raise SystemExit("pairs.json is not a list")
    by_site: dict[str, dict] = {}
    for pair in pairs:
        site_id = str(pair.get("site_id") or "")
        if site_id not in deep_sites:
            audit(rejected, "effects_site_not_in_registry", site_id=site_id)
            continue
        if site_id in by_site:
            audit(rejected, "duplicate_effects_site", site_id=site_id)
            continue
        if pair.get("text_stored") is not False:
            audit(rejected, "effects_text_stored_unexpected", site_id=site_id)
            continue
        operators = []
        blobs = []
        for group in ("feature_checks", "extra_operators"):
            raw = pair.get(group) or {}
            if not isinstance(raw, dict):
                audit(rejected, "effects_operator_group_invalid", site_id=site_id, group=group)
                continue
            for name, rec in raw.items():
                if not isinstance(rec, dict):
                    audit(rejected, "effects_operator_not_object", site_id=site_id, operator=name)
                    continue
                logit_l2 = finite_number(rec.get("logit_l2"))
                if logit_l2 is None or logit_l2 < 0:
                    audit(rejected, "effects_operator_missing_logit_l2", site_id=site_id, operator=name)
                    continue
                feature_ids = []
                if isinstance(rec.get("feature_ids"), list):
                    feature_ids = [int(item) for item in rec["feature_ids"]]
                elif "feature_id" in rec:
                    feature_ids = [int(rec["feature_id"])]
                parameters = {}
                if "single_token_position" in rec:
                    parameters["single_token_position"] = bool(rec["single_token_position"])
                operators.append(
                    {
                        "feature_ids": feature_ids,
                        "logit_l2": logit_l2,
                        "operation": str(name),
                        "operation_parameters": parameters,
                    }
                )
                blobs.append(name)
        if not operators:
            audit(rejected, "effects_site_without_operators", site_id=site_id)
            continue
        order = {name: index for index, name in enumerate(OPERATOR_ORDER)}
        operators.sort(key=lambda item: (order.get(item["operation"], 99), item["operation"]))
        primary = next((item for item in operators if item["operation"] == "ablate"), operators[0])
        model, sae, layer = parse_site(site_id)
        prompt_group_id = str(pair.get("prompt_group_id") or "")
        case_id = f"gpu04:{site_id}:{prompt_group_id}"
        feature_ids = sorted({feature for item in operators for feature in item["feature_ids"]})
        by_site[site_id] = {
            "case_id": case_id,
            "condition": primary["operation"],
            "evidence_url": EFFECTS_URL,
            "feature_id": primary["feature_ids"][0],
            "feature_ids": feature_ids,
            "id": case_id,
            "identity_patch_passed": pair.get("identity_patch") == "passed",
            "joint_multi_token_patch": bool(pair.get("joint_multi_token_patch")),
            "layer": layer,
            "measured_delta_l2": primary["logit_l2"],
            "measured_fields": ["logit_l2"],
            "model": model,
            "model_sha256": str(pair.get("loaded_model_sha256") or ""),
            "note": "Logit L2 is the recorded distance between next-token logit vectors. This pair does not store token probabilities or a generated sentence.",
            "observation_position": "single_token_position" if primary["operation_parameters"].get("single_token_position") else "recorded_readout",
            "operators": operators,
            "prompt_group_id": prompt_group_id,
            "protocol_id": EFFECTS_PROTOCOL,
            "public_text_reviewed": False,
            "release_schema": EFFECTS_SCHEMA,
            "rights_reviewed": False,
            "run_id": RUN_ID,
            "sae": sae,
            "sae_sha256": str(pair.get("loaded_sae_sha256") or ""),
            "sample_count": 1,
            "site_id": site_id,
            "source_category": prompt_group_id.split(":", 1)[0],
            "source_relative_path": "GPU-04/interventions/pairs.json",
            "source_sha256": source_sha,
            "source_type": "real_checkpoint",
            "text_stored": False,
            "title": f"{model} L{layer} · feature {primary['feature_ids'][0]}",
        }
    missing = [site for site in deep_sites if site not in by_site]
    if missing:
        raise SystemExit("GPU-04 is missing recorded pairs for: " + ", ".join(missing))
    cases = [by_site[site] for site in deep_sites]
    generation_path = run_root / "GPU-04" / "generation_v2.json"
    generation_note = {"present": generation_path.is_file(), "joined_into_effects": False}
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        generation_note.update(
            {
                "sample_id": generation.get("sample_id"),
                "site_id": generation.get("site_id"),
                "status": generation.get("status"),
                "token_ids_only": isinstance(generation.get("baseline_top5"), list)
                and all(isinstance(item, int) for item in generation.get("baseline_top5") or []),
                "reason": "Token ids without probabilities, and a different sample id from the six pairs, are not a probability chart.",
            }
        )
    payload = {
        "cases": cases,
        "release_schema": EFFECTS_SCHEMA,
        "run_id": RUN_ID,
        "source_population": "Six deep sites, one recorded prompt group each, every stored operator on that pair.",
    }
    return payload, {"source_sha256": source_sha, "n_cases": len(cases), "generation_v2": generation_note}


def _mask_from_bits(bits: list[int]) -> int:
    if len(bits) != 6 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must be six binary flags")
    return sum(bit << index for index, bit in enumerate(bits))


def build_context(run_root: Path, rejected: list[dict]) -> tuple[dict, dict]:
    design_path = run_root / "GPU-07" / "real_model_six_factor_design.json"
    codes_path = run_root / "private" / CODES_NAME
    if not design_path.is_file() or not codes_path.is_file():
        raise SystemExit("Missing six-factor design or Pythia L12 codes")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("schema") != CONTEXT_PROTOCOL:
        raise SystemExit(f"Unexpected design schema: {design.get('schema')}")
    if design.get("state_order") != "integer_bitmask_0_to_63_factor_zero_least_significant":
        raise SystemExit("Design bit order is not factor-zero least significant")
    if design.get("source_type") != "project_authored_controlled_input":
        raise SystemExit("Refusing a six-factor design that is not project-authored")
    states = design.get("states")
    factors = design.get("factors")
    if not isinstance(states, list) or len(states) != 64 or not isinstance(factors, list) or len(factors) != 6:
        raise SystemExit("Design is not a six-factor 64-state grid")
    by_sample: dict[str, dict] = {}
    prompts = []
    for state in states:
        mask = int(state["mask"])
        bits = [int(bit) for bit in state["bits"]]
        if _mask_from_bits(bits) != mask or not 0 <= mask <= 63:
            raise SystemExit(f"Design mask/bits mismatch at {state.get('sample_id')}")
        sample_id = str(state["sample_id"])
        if sample_id in by_sample:
            raise SystemExit(f"Duplicate design sample id {sample_id}")
        prompt = state.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"Design state {sample_id} has no authored prompt")
        by_sample[sample_id] = state
        prompts.append({"mask": mask, "prompt": prompt, "sample_id": sample_id})
    if len({item["mask"] for item in prompts}) != 64:
        raise SystemExit("Design masks are not a permutation of 0..63")
    prompts.sort(key=lambda item: item["mask"])
    observed: dict[str, dict[int, float]] = {}
    hashes: set[tuple[str, str]] = set()
    positions: dict[str, int] = {}
    with codes_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset_id") != "controlled6":
                continue
            sample_id = str(row.get("sample_id") or "")
            if sample_id not in by_sample:
                audit(rejected, "controlled_code_sample_not_in_design", sample_id=sample_id, line=line_number)
                continue
            if row.get("site_id") != CONTEXT_SITE:
                raise SystemExit(f"Controlled code site mismatch at {sample_id}")
            if sample_id in observed:
                audit(rejected, "duplicate_controlled_code", sample_id=sample_id)
                continue
            mapping: dict[int, float] = {}
            for pair in row.get("nonzero_codes") or []:
                feature_id = int(pair[0])
                value = finite_number(pair[1])
                if value is None:
                    raise SystemExit(f"Non-finite code at {sample_id}")
                if feature_id in mapping:
                    audit(rejected, "duplicate_feature_code_in_row", sample_id=sample_id, feature_id=feature_id)
                    continue
                mapping[feature_id] = value
            observed[sample_id] = mapping
            hashes.add((str(row.get("loaded_model_sha256") or ""), str(row.get("loaded_sae_sha256") or "")))
            position = row.get("token_position")
            if isinstance(position, int):
                positions[sample_id] = position
    missing = sorted(set(by_sample) - set(observed))
    if missing:
        raise SystemExit(f"{len(missing)} design states have no SAE codes; refusing to fill them")
    if len(hashes) != 1:
        raise SystemExit("Controlled rows do not share one checkpoint identity")
    model_sha, sae_sha = next(iter(hashes))
    feature_ids = sorted({feature for mapping in observed.values() for feature in mapping})
    model, sae, layer = parse_site(CONTEXT_SITE)
    public_factors = []
    for index, factor in enumerate(factors):
        levels = factor.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            raise SystemExit("Each factor needs two authored levels")
        public_factors.append({"id": str(factor["name"]), "index": index, "label": str(factor["name"]), "levels": [str(levels[0]), str(levels[1])]})
    features = []
    for feature_id in feature_ids:
        activations = []
        for prompt in prompts:
            activations.append(float(observed[prompt["sample_id"]].get(feature_id, 0.0)))
        if len(activations) != 64 or any(not math.isfinite(value) for value in activations):
            raise SystemExit(f"Feature {feature_id} is not a finite 64-vector")
        mean = sum(activations) / 64
        variance = sum((value - mean) ** 2 for value in activations) / 64
        marginals = []
        for bit in range(6):
            diffs = [activations[mask] - activations[mask ^ (1 << bit)] for mask in range(64) if mask & (1 << bit)]
            if len(diffs) != 32:
                raise SystemExit("Paired contrast is not 32 rows")
            marginals.append(
                {
                    "factor": public_factors[bit]["label"],
                    "paired_mean_difference": sum(diffs) / 32,
                }
            )
        features.append(
            {
                "activations": activations,
                "feature_id": feature_id,
                "marginals": marginals,
                "max": max(activations),
                "mean": mean,
                "nonzero_count": sum(value != 0 for value in activations),
                "variance": variance,
            }
        )
    features.sort(key=lambda item: (-item["variance"], item["feature_id"]))
    for rank, feature in enumerate(features, start=1):
        feature["rank"] = rank
        feature["easy"] = rank <= 12 or feature["variance"] > 0
    payload = {
        "design_name": "real_model_six_factor_design",
        "factors": public_factors,
        "feature_ids": [item["feature_id"] for item in features],
        "features": features,
        "independent_group_count": 1,
        "model": model,
        "model_sha256": model_sha,
        "prompts": prompts,
        "protocol_id": CONTEXT_PROTOCOL,
        "public_text_reviewed": True,
        "release_schema": CONTEXT_SCHEMA,
        "rights_reviewed": False,
        "run_id": RUN_ID,
        "sae": sae,
        "sae_sha256": sae_sha,
        "scope": "One 64-state project-authored population. Factor differences describe this grid only.",
        "site_id": CONTEXT_SITE,
        "source_relative_path": f"private/{CODES_NAME}",
        "source_sha256": sha256_file(codes_path),
        "source_type": "real_checkpoint",
        "design_sha256": sha256_file(design_path),
        "token_positions": {prompts[mask]["sample_id"]: positions[prompts[mask]["sample_id"]] for mask in range(64) if prompts[mask]["sample_id"] in positions},
    }
    summary = {
        "n_states": 64,
        "n_features": len(features),
        "codes_sha256": payload["source_sha256"],
        "design_sha256": payload["design_sha256"],
        "model_sha256": model_sha,
        "sae_sha256": sae_sha,
    }
    return payload, summary


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def build_rss(run_root: Path, deep_sites: list[str], model_hashes: dict[str, tuple[str, str]], rejected: list[dict]) -> tuple[dict, dict]:
    root = run_root / "GPU-06" / "rss_v2"
    if not root.is_dir():
        raise SystemExit(f"Missing position-corrected RSS store: {root}")
    legacy_root = run_root / "GPU-06" / "rss"
    legacy_counts = []
    if legacy_root.is_dir():
        for path in sorted(legacy_root.glob("*.jsonl")):
            if path.name.endswith("_invalid.jsonl"):
                audit(rejected, "invalid_rss_jsonl_not_measured", path=str(path.relative_to(run_root)))
                legacy_counts.append({"path": str(path.relative_to(run_root)), "rows": None, "invalid": True})
                continue
            settings: dict[str, int] = defaultdict(int)
            position_flags: dict[str, int] = defaultdict(int)
            rows = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows += 1
                    row = json.loads(line)
                    for setting in row.get("settings") or []:
                        if not isinstance(setting, dict):
                            continue
                        settings[str(setting.get("setting"))] += 1
                        position_flags[str(setting.get("position_recomputed"))] += 1
            legacy_counts.append({"path": str(path.relative_to(run_root)), "rows": rows, "settings": dict(settings), "position_recomputed": dict(position_flags)})
            audit(
                rejected,
                "legacy_or_unverified_rss_jsonl_not_pooled",
                path=str(path.relative_to(run_root)),
                rows=rows,
                position_recomputed=dict(position_flags),
            )
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    file_hashes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    seen_samples: set[tuple[str, str, str, str]] = set()
    counts = {"files": 0, "defined": 0, "undefined": 0, "other": 0}
    for site_id in deep_sites:
        folder = root / site_id.replace(":", "_")
        if not folder.is_dir():
            raise SystemExit(f"Missing RSS directory for {site_id}")
        for path in sorted(folder.glob("*.json")):
            counts["files"] += 1
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("position_recomputed") is not True:
                audit(rejected, "rss_row_position_not_recomputed", path=str(path.relative_to(run_root)))
                continue
            dataset_id = str(row.get("dataset_id") or "")
            sample_id = str(row.get("sample_id") or "")
            if not dataset_id or not sample_id:
                audit(rejected, "rss_row_missing_ids", path=str(path.relative_to(run_root)))
                continue
            file_sha = sha256_file(path)
            settings = row.get("settings") or {}
            for setting in SETTING_ORDER:
                rec = settings.get(setting)
                if not isinstance(rec, dict):
                    audit(rejected, "rss_setting_absent", site_id=site_id, dataset_id=dataset_id, setting=setting, sample_id=sample_id)
                    counts["other"] += 1
                    continue
                status = rec.get("status")
                key = (site_id, dataset_id, setting)
                marker = (site_id, dataset_id, setting, sample_id)
                if marker in seen_samples:
                    audit(rejected, "duplicate_rss_sample", site_id=site_id, dataset_id=dataset_id, setting=setting, sample_id=sample_id)
                    continue
                seen_samples.add(marker)
                if status == "undefined":
                    counts["undefined"] += 1
                    grouped[key].append({"sample_id": sample_id, "status": "undefined"})
                    file_hashes[key].append(file_sha)
                    continue
                if status != "measured":
                    counts["other"] += 1
                    audit(rejected, "rss_status_not_measured", site_id=site_id, setting=setting, sample_id=sample_id, status=status)
                    continue
                mean_rss = finite_number(rec.get("mean_rss"))
                held = finite_number(rec.get("held_out_rate_mean"))
                support = rec.get("support")
                if mean_rss is None or held is None or not isinstance(support, list):
                    audit(rejected, "rss_measured_row_incomplete", site_id=site_id, setting=setting, sample_id=sample_id)
                    continue
                if any(not isinstance(feature, int) for feature in support):
                    audit(rejected, "rss_support_not_ints", site_id=site_id, sample_id=sample_id, setting=setting)
                    continue
                counts["defined"] += 1
                grouped[key].append(
                    {
                        "held_out_rate_mean": held,
                        "held_out_variant_count": len(rec.get("held_out_variant_ids") or []),
                        "mean_rss": mean_rss,
                        "sample_id": sample_id,
                        "selection_variant_count": len(rec.get("selection_variant_ids") or []),
                        "status": "measured",
                        "support": sorted(int(feature) for feature in support),
                        "support_count": len(support),
                    }
                )
                file_hashes[key].append(file_sha)
    cases = []
    for site_id in deep_sites:
        model, sae, layer = parse_site(site_id)
        datasets = sorted({dataset for (site, dataset, _setting) in grouped if site == site_id})
        for dataset_id in datasets:
            for setting in SETTING_ORDER:
                rows = grouped.get((site_id, dataset_id, setting), [])
                defined = [row for row in rows if row["status"] == "measured"]
                undefined = [row for row in rows if row["status"] == "undefined"]
                if not defined:
                    audit(rejected, "rss_aggregate_without_defined_rows", site_id=site_id, dataset_id=dataset_id, setting=setting, n_undefined=len(undefined))
                    continue
                means = [row["mean_rss"] for row in defined]
                helds = [row["held_out_rate_mean"] for row in defined]
                supports = [row["support_count"] for row in defined]
                digest = hashlib.sha256()
                for file_sha in sorted(file_hashes[(site_id, dataset_id, setting)]):
                    digest.update(file_sha.encode("ascii"))
                model_sha, sae_sha = model_hashes.get(site_id, ("", ""))
                variants = []
                for row in sorted(defined, key=lambda item: item["sample_id"]):
                    variants.append(
                        {
                            "held_out_rate_mean": row["held_out_rate_mean"],
                            "held_out_variant_count": row["held_out_variant_count"],
                            "id": row["sample_id"],
                            "label": row["sample_id"],
                            "mean_rss": row["mean_rss"],
                            "selection_variant_count": row["selection_variant_count"],
                            "split": "measured",
                            "support": row["support"],
                            "support_count": row["support_count"],
                        }
                    )
                cases.append(
                    {
                        "dataset_id": dataset_id,
                        "evidence_url": RSS_URL,
                        "held_out_rate_mean": _mean(helds),
                        "held_out_variant_count": sum(row["held_out_variant_count"] for row in defined),
                        "id": f"{site_id}|{dataset_id}|{setting}",
                        "label": "proposed transformation",
                        "layer": layer,
                        "mean_rss": _mean(means),
                        "mean_rss_max": max(means),
                        "mean_rss_min": min(means),
                        "measured_fields": ["mean_rss", "held_out_rate_mean", "support"],
                        "model": model,
                        "model_sha256": model_sha,
                        "n_defined": len(defined),
                        "n_parent": len(defined) + len(undefined),
                        "n_undefined": len(undefined),
                        "parent_mean_rss_std": statistics.pstdev(means) if len(means) > 1 else 0.0,
                        "protocol_id": RSS_PROTOCOL,
                        "public_text_reviewed": False,
                        "release_schema": RSS_SCHEMA,
                        "rights_reviewed": False,
                        "run_id": RUN_ID,
                        "sae": sae,
                        "sae_sha256": sae_sha,
                        "selection_variant_count": sum(row["selection_variant_count"] for row in defined),
                        "semantic_review": False,
                        "setting": setting,
                        "site_id": site_id,
                        "source_population": "Position-corrected parent groups. Undefined supports stay out of the mean.",
                        "source_relative_path": f"GPU-06/rss_v2/{site_id.replace(':', '_')}",
                        "source_sha256": digest.hexdigest(),
                        "source_type": "real_checkpoint",
                        "support_count_max": max(supports),
                        "support_count_mean": _mean([float(value) for value in supports]),
                        "support_count_min": min(supports),
                        "title": f"{model} L{layer} · {dataset_id} · {setting}",
                        "variants": variants,
                    }
                )
    if not cases:
        raise SystemExit("RSS export produced no measured aggregates")
    payload = {
        "cases": cases,
        "release_schema": RSS_SCHEMA,
        "run_id": RUN_ID,
        "source_population": "Position-corrected RSS parent groups for the six deep sites. Legacy JSONL rows are excluded.",
        "wording": "proposed transformation",
    }
    summary = {"counts": counts, "n_aggregates": len(cases), "legacy_jsonl": legacy_counts}
    return payload, summary


def lab_contexts(context: dict) -> list[dict]:
    prompts = {item["mask"]: item for item in context["prompts"]}
    records = []
    for feature in context["features"]:
        states = []
        for mask, activation in enumerate(feature["activations"]):
            prompt = prompts[mask]
            states.append(
                {
                    "activation": activation,
                    "mask": mask,
                    "prompt": prompt["prompt"],
                    "sample_id": prompt["sample_id"],
                }
            )
        records.append(
            {
                "evidence_url": CONTEXT_URL,
                "factors": context["factors"],
                "feature_id": feature["feature_id"],
                "id": f"controlled6:{context['site_id']}:{feature['feature_id']}",
                "layer": parse_site(context["site_id"])[2],
                "marginals": feature["marginals"],
                "model": context["model"],
                "model_sha256": context["model_sha256"],
                "nonzero_count": feature["nonzero_count"],
                "protocol_id": CONTEXT_PROTOCOL,
                "public_text_reviewed": True,
                "rank": feature["rank"],
                "rights_reviewed": False,
                "run_id": RUN_ID,
                "sae": context["sae"],
                "sae_sha256": context["sae_sha256"],
                "scope": context["scope"],
                "site_id": context["site_id"],
                "source_relative_path": context["source_relative_path"],
                "source_sha256": context["source_sha256"],
                "source_type": "real_checkpoint",
                "states": states,
                "title": f"Feature {feature['feature_id']}",
                "variance": feature["variance"],
            }
        )
    return records


def build_lab_pack(effects: dict, context: dict, rss: dict, prior_pack: Path | None) -> dict:
    sites: list = []
    metrics: list = []
    coalitions: list = []
    if prior_pack and prior_pack.is_file():
        prior = json.loads(prior_pack.read_text(encoding="utf-8"))
        sites = prior.get("sites") or []
        metrics = prior.get("metrics") or []
        coalitions = prior.get("coalitions") or []
    return {
        "coalitions": coalitions,
        "contexts": lab_contexts(context),
        "effects": effects["cases"],
        "meta": {
            "run_id": RUN_ID,
            "source": EFFECTS_URL,
            "note": "Effects, controlled inputs, and input transformations are preserved measurements.",
        },
        "metrics": metrics,
        "rss": rss["cases"],
        "schema": LAB_SCHEMA,
        "sites": sites,
    }


def inventory(run_root: Path, effects_meta: dict, context_meta: dict, rss_meta: dict, effects: dict, context: dict, rss: dict) -> dict:
    localize = run_root / "GPU-07" / "localize_status.json"
    localize_note = {"present": localize.is_file(), "used_as_matrix": False}
    return {
        "context": context_meta,
        "effects": {
            **effects_meta,
            "by_site": [
                {
                    "feature_ids": case["feature_ids"],
                    "logit_l2_only": case["measured_fields"] == ["logit_l2"],
                    "operators": [item["operation"] for item in case["operators"]],
                    "prompt_group_id": case["prompt_group_id"],
                    "site_id": case["site_id"],
                    "top_tokens": False,
                }
                for case in effects["cases"]
            ],
        },
        "gpu01_top_tokens_joined": False,
        "localize_status": localize_note,
        "optional_gpu_followup": False,
        "rss": {
            "aggregates": rss_meta["n_aggregates"],
            "by_site_setting": _rss_census(rss),
            "counts": rss_meta["counts"],
            "legacy_jsonl_excluded": rss_meta["legacy_jsonl"],
        },
        "run_id": RUN_ID,
        "run_root_name": run_root.name,
        "v07_gap": "The deployed experiment pack had no 64-state grid, no RSS rows, and effect rows without operator series, so those screens had nothing measured to draw.",
    }


def _rss_census(rss: dict) -> list[dict]:
    rows = []
    for case in rss["cases"]:
        rows.append(
            {
                "dataset_id": case["dataset_id"],
                "mean_rss": case["mean_rss"],
                "n_defined": case["n_defined"],
                "n_undefined": case["n_undefined"],
                "setting": case["setting"],
                "site_id": case["site_id"],
            }
        )
    return rows


def manifest_for(files: dict[str, str], effects: dict, context: dict, rss: dict) -> dict:
    return {
        "assets": files,
        "context_features": len(context["features"]),
        "context_states": 64,
        "effects_cases": len(effects["cases"]),
        "release_schema": MANIFEST_SCHEMA,
        "rss_aggregates": len(rss["cases"]),
        "run_id": RUN_ID,
        "source_population": {
            "context": context["scope"],
            "effects": effects["source_population"],
            "rss": rss["source_population"],
        },
    }


def build(run_root: Path, registry: Path, out: Path, prior_pack: Path | None, lab_evidence: Path | None, publish: bool = True) -> dict:
    rejected: list[dict] = []
    deep_sites = load_registry(registry)
    effects, effects_meta = build_effects(run_root, deep_sites, rejected)
    context, context_meta = build_context(run_root, rejected)
    hashes = {case["site_id"]: (case["model_sha256"], case["sae_sha256"]) for case in effects["cases"]}
    rss, rss_meta = build_rss(run_root, deep_sites, hashes, rejected)
    report = inventory(run_root, effects_meta, context_meta, rss_meta, effects, context, rss)
    report["rejected"] = len(rejected)
    if not publish:
        return {"effects": effects, "context": context, "rss": rss, "manifest": None, "rejected": rejected, "inventory": report, "pack": None}
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "context64.json": write_json(out / "context64.json", context),
        "effects.json": write_json(out / "effects.json", effects),
        "rss.json": write_json(out / "rss.json", rss),
    }
    manifest = manifest_for(written, effects, context, rss)
    write_json(out / "manifest.json", manifest)
    pack = build_lab_pack(effects, context, rss, prior_pack)
    if lab_evidence:
        write_json(lab_evidence, pack)
    links = []
    write_json(out / "effect-links.json", links)
    return {"effects": effects, "context": context, "rss": rss, "manifest": manifest, "rejected": rejected, "inventory": report, "pack": pack}


def verify(run_root: Path, registry: Path, out: Path, prior_pack: Path | None) -> None:
    fresh = build(run_root, registry, out / "_verify_tmp", prior_pack, None)
    for name in ("effects.json", "context64.json", "rss.json"):
        current = (out / name).read_text(encoding="utf-8")
        expected = dump_json(fresh[{"effects.json": "effects", "context64.json": "context", "rss.json": "rss"}[name]])
        if current != expected:
            raise SystemExit(f"Verify failed: {name} is not a deterministic rebuild")
    tmp = out / "_verify_tmp"
    if tmp.exists():
        for child in tmp.iterdir():
            child.unlink()
        tmp.rmdir()


def write_inventory(path: Path, payload: dict, rejected: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    markdown = path.with_suffix(".md")
    lines = [
        "# Recorded measurement inventory",
        "",
        f"Run `{payload['run_id']}`.",
        "",
        f"Feature Effects cases: {payload['effects']['n_cases']}. Each case stores logit L2 only.",
        f"Controlled states with codes: {payload['context']['n_states']}. Observed-support features: {payload['context']['n_features']}.",
        f"RSS aggregates: {payload['rss']['aggregates']}. Defined setting rows: {payload['rss']['counts']['defined']}. Undefined setting rows: {payload['rss']['counts']['undefined']}.",
        "",
        "Legacy RSS JSONL files were counted and left out of the position-corrected aggregates.",
        "No optional GPU remeasurement was run.",
        "",
    ]
    markdown.write_text("\n".join(lines), encoding="utf-8")
    audit_path = path.parent / "rejected.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export recorded Research Lab cases")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--site-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-pack", type=Path)
    parser.add_argument("--lab-evidence", type=Path)
    parser.add_argument("--inventory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.run_root.is_dir():
        raise SystemExit(f"Run root not found: {args.run_root}")
    if args.verify:
        verify(args.run_root, args.site_registry, args.out, args.prior_pack)
        print("verify ok")
        return
    result = build(args.run_root, args.site_registry, args.out, args.prior_pack, args.lab_evidence if args.build else None, publish=args.build)
    if args.inventory:
        write_inventory(args.inventory, result["inventory"], result["rejected"])
    print(
        json.dumps(
            {
                "context_features": result["inventory"]["context"]["n_features"],
                "effects": result["inventory"]["effects"]["n_cases"],
                "rejected": result["inventory"]["rejected"],
                "rss_aggregates": result["inventory"]["rss"]["aggregates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
