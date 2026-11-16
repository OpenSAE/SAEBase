"""SAR-only to_ziye v2 filtering before candidate selection and coalition fitting.

Rules originate from SAR/to_ziye's filter_sae_features.py. Feature IDs and
native activation densities always belong to the currently loaded SAE.
"""

from importlib.resources import files
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

POLICY_PATH = Path(str(files("conf") / "task/sar_filter_policy.json"))
POLICY = json.loads(POLICY_PATH.read_text())
PUNCTUATION_RULES = POLICY["punctuation_rules"]
STRUCTURAL_RULES = POLICY["structural_rules"]
FUNCTION_WORDS = set(POLICY["named_function_words"])
PROGRAMMING_CONTEXT = re.compile(
    r"\b(?:programming|code|coding|software|sql|javascript|python|html|css|php|java|typescript)\b",
    re.I,
)


def structural_matches(description):
    """Pure linguistic form; do not confuse programming syntax or topical nouns with it."""
    text = description.casefold()
    hits = []
    for rule, pattern in STRUCTURAL_RULES.items():
        if rule in (
            "natural_language_grammar",
            "sentence_structure",
            "function_word_class",
            "verb_morphology",
        ) and PROGRAMMING_CONTEXT.search(text):
            continue
        if re.search(pattern, text):
            hits.append(rule)
    # Match a named function word, not any occurrence of that word in a label.
    for match in re.finditer(
        r"\b(?:words?|tokens?)\s+[\"\u201c\u2018\x27`]([^\"\u201d\u2019\x27`]+)[\"\u201d\u2019\x27`]",
        description,
        re.I,
    ):
        word = match.group(1).strip().casefold().strip(".,;:!?")
        if word not in FUNCTION_WORDS:
            continue
        if word == "may" and re.search(r"\b(?:month|calendar|spring|date)\b", text):
            continue
        if word == "will" and re.search(
            r"\b(?:person|actor|surname|first name|proper name)\b", text
        ):
            continue
        hits.append("named_function_word:" + word)
    return sorted(set(hits))


def punctuation_matches(description):
    text = description.casefold()
    hits = []
    for rule, pattern in PUNCTUATION_RULES.items():
        if not re.search(pattern, text):
            continue
        if (
            rule == "colon_semicolon"
            and re.search(r"\b(?:cancer|bowel|intestin\w*|colorectal|digestive)\b", text)
            and not re.search(r"\b(?:punctuation|semicolon|syntax)\b", text)
        ):
            continue
        hits.append(rule)
    return hits


def filter_settings(raw):
    if not hasattr(raw, "get"):
        raise ValueError("task.sar.feature_filter must contain filter settings")
    enabled = raw.get("enabled", False)  # Historical explicit configs retain their protocol.
    threshold = raw.get("min_native_nonzero_fraction", 0.5)
    if type(enabled) is not bool:
        raise ValueError("task.sar.feature_filter.enabled must be boolean")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("SAR native nonzero fraction must be between zero and one")
    return {"enabled": enabled, "min_native_nonzero_fraction": float(threshold)}


def blocked_features(descriptions, densities, threshold):
    """Preserve every matching rule and the actual density for each excluded ID."""
    blocked = {}
    for fid, annotation in descriptions.items():
        reasons = set()
        for item in annotation.get("explanations", []):
            text = item["description"]
            reasons.update("punctuation:" + hit for hit in punctuation_matches(text))
            reasons.update("structure:" + hit for hit in structural_matches(text))
        density = densities[str(fid)]
        if density is not None and density >= threshold:
            reasons.add("widespread_native_activation")
        if reasons:
            blocked[fid] = {"rules": sorted(reasons), "native_nonzero_fraction": density}
    return blocked


def cohort_filter_settings(raw):
    """Validate question-frequency thresholds independently of semantic filtering."""
    if not hasattr(raw, "get"):
        raise ValueError("task.sar.cohort_filter must contain settings")
    enabled = raw.get("enabled", False)
    threshold = raw.get("zscore_threshold", 1.0)
    fraction = raw.get("min_question_fraction", 0.5)
    reference_count = raw.get("reference_sample_count")
    if type(enabled) is not bool:
        raise ValueError("task.sar.cohort_filter.enabled must be boolean")
    if reference_count is not None and (type(reference_count) is not int or reference_count < 1):
        raise ValueError("SAR cohort reference_sample_count must be a positive integer or null")
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        raise ValueError("SAR cohort zscore_threshold must be finite")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("SAR cohort min_question_fraction must be in (0, 1]")
    return {
        "enabled": enabled,
        "zscore_threshold": float(threshold),
        "min_question_fraction": float(fraction),
        "reference_sample_count": reference_count,
    }


def prepare_cohort_filter(context, normalization, snapshots, rows):
    """Freeze an all-question last-token frequency mask before any SAR selection."""
    from saebase.runtime.experiment_results import write_json

    settings = context["settings"]["cohort_filter"]
    if not settings["enabled"]:
        return
    if not rows or len(snapshots) != len(rows):
        raise ValueError("SAR cohort filtering requires snapshots for every selected question")
    counts = np.zeros(len(normalization.mean), dtype=np.int64)
    content = hashlib.sha256()
    for row, snapshot in zip(rows, snapshots):
        scores, _ = normalization.rank(snapshot.codes, 0)
        counts += scores > settings["zscore_threshold"]
        content.update(
            json.dumps(
                {"sample_id": row["sample_id"], "prompt": row["prompt"]},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        )
        content.update(snapshot.codes.detach().cpu().float().numpy().astype("<f4").tobytes())
    blocked = np.flatnonzero(counts / len(rows) >= settings["min_question_fraction"]).tolist()
    identity = {
        **settings,
        "question_count": len(rows),
        "blocked_count": len(blocked),
        "cohort_sha256": content.hexdigest(),
        "normalization": normalization.identity,
        "count_comparison": "signed zscore > threshold",
        "frequency_comparison": "count / all_selected_questions >= min_question_fraction",
        "reference_selection": context.get("reference_selection"),
    }
    path = Path.cwd() / "metadata/sar_cohort_filter.json"
    write_json(
        path,
        {
            "identity": identity,
            "sample_ids": [r["sample_id"] for r in rows],
            "exceedance_counts": counts.tolist(),
            "blocked_ids": blocked,
        },
    )
    context.update(
        cohort_blocked_features=set(blocked),
        cohort_filter_identity={
            **identity,
            "artifact": {
                "path": "metadata/sar_cohort_filter.json",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        },
    )


def select_features(pair, codes, normalization, count, context, options):
    """Filter the complete ranking before Top K; blocked endpoints never enter fits."""
    if count > codes.numel():
        raise ValueError(f"K={count} exceeds SAE feature count {codes.numel()}")
    settings = context["settings"]["feature_filter"]
    if settings["enabled"] and "blocked_features" not in context:
        from saebase.api.neuronpedia import annotate_features, feature_activation_densities
        from saebase.runtime.experiment_results import write_json

        annotation, descriptions = annotate_features(
            pair, list(range(codes.numel())), {**options, "enabled": True}
        )
        if annotation["status"] not in {"complete", "partial"}:
            raise ValueError(
                "SAR feature filtering requires verified Neuronpedia explanations: "
                + annotation.get("reason", "")
            )
        densities = feature_activation_densities(pair, options)
        blocked = blocked_features(
            descriptions, densities["fractions"], settings["min_native_nonzero_fraction"]
        )
        identity = {
            **settings,
            "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            "neuronpedia_id": annotation["neuronpedia_id"],
            "explanations_sha256": annotation["export_sha256"],
            "densities_sha256": densities["sha256"],
            "blocked_count": len(blocked),
            "unexplained_count": sum(not v["explanations"] for v in descriptions.values()),
            "unknown_density_count": sum(v is None for v in densities["fractions"].values()),
            "missing_metadata_policy": "apply available evidence; retain unknown labels/fractions",
        }
        path = Path.cwd() / "metadata/sar_feature_filter.json"
        write_json(
            path,
            {
                "identity": identity,
                "policy": POLICY,
                "blocked_features": blocked,
                "density_provenance": densities["provenance"],
            },
        )
        context.update(
            blocked_features=blocked,
            feature_filter_identity={
                **identity,
                "artifact": {
                    "path": "metadata/sar_feature_filter.json",
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            },
        )
    if normalization is None:
        values = codes.detach().cpu().numpy()
        ranking = np.lexsort((np.arange(len(values)), -np.abs(values))).tolist()
    else:
        _, ranking = normalization.rank(codes, codes.numel())
    if (
        context["settings"].get("cohort_filter", {}).get("enabled")
        and "cohort_blocked_features" not in context
    ):
        raise ValueError(
            "SAR cohort filter must be prepared from all selected questions before selection"
        )
    blocked = set(context.get("blocked_features", {})) | context.get(
        "cohort_blocked_features", set()
    )
    selected = [i for i in ranking if i not in blocked][:count]
    if len(selected) != count:
        raise ValueError(
            f"SAR has only {len(selected)} eligible features after filtering; requires {count}"
        )
    return selected
