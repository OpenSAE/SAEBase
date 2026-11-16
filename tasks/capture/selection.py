"""Capture-owned candidate filtering and independent observation limits."""

from tasks.capture.sar_filter import filter_settings, cohort_filter_settings


def settings(cfg):
    task = cfg.get("task", {})
    modern_mode = task.get("mode") in {"g_s", "I"}
    explicit = "filter" in task or modern_mode
    limit = task.get("filter", 10 if modern_mode else "all")
    if limit != "all" and (type(limit) is not int or limit < 1):
        raise ValueError("task.filter must be 'all' or a positive integer")
    count = task.get("capture_count")
    legacy = task.get("top_feature_num")
    if count is None:
        count = legacy if legacy is not None else (limit if explicit else 10)
    if count != "all" and (type(count) is not int or count < 1):
        raise ValueError("task.capture_count must be 'all' or a positive integer")
    if (
        task.get("mode") not in {"sar", "I"}
        and type(limit) is int
        and type(count) is int
        and count > limit
    ):
        raise ValueError("task.capture_count cannot exceed the task.filter candidate count")
    return {"filter": limit, "capture_count": count, "explicit": explicit}


def output_count(cfg, available):
    value = settings(cfg)["capture_count"]
    return available if value == "all" else min(value, available)


def prepare_filter(cfg):
    selection = settings(cfg)
    if selection["filter"] == "all":
        return None
    raw = cfg.task.get("sar", {})
    feature = dict(raw.get("feature_filter", {}))
    feature.setdefault("enabled", True)
    cohort = dict(raw.get("cohort_filter", {}))
    cohort.setdefault("enabled", True)
    cohort.setdefault("reference_sample_count", 100)
    return {
        "settings": {
            "feature_filter": filter_settings(feature),
            "cohort_filter": cohort_filter_settings(cohort),
        }
    }


def needs_normalization(cfg, context=None):
    task = cfg.get("task", {})
    mode = task.get("mode", "non_whiten")
    default_selection = "whiten" if mode in {"g_s", "I"} else "native_activation"
    all_behavior = (
        settings(cfg)["explicit"]
        and settings(cfg)["filter"] == "all"
        and mode in {"g_s", "sar", "I"}
    )
    return (
        mode == "whiten"
        or (
            not all_behavior
            and mode in {"g_s", "sar", "I"}
            and task.get("sar", {}).get("feature_selection", default_selection) == "whiten"
        )
        or (context is not None and context["settings"]["cohort_filter"]["enabled"])
    )


def candidate_ids(cfg, pair, codes, normalization, context, options):
    import numpy as np

    value = settings(cfg)["filter"]
    mode = cfg.task.get("mode", "non_whiten")
    if value == "all" and mode in {"g_s", "sar", "I"}:
        return list(range(codes.numel()))
    whiten = mode == "whiten" or (
        mode in {"sar", "g_s", "I"}
        and cfg.task.get("sar", {}).get("feature_selection", "whiten") == "whiten"
    )
    ranking_norm = normalization if whiten else None
    if context is not None:
        from tasks.capture.sar_filter import select_features

        count = value if type(value) is int else codes.numel()
        return select_features(pair, codes, ranking_norm, count, context, options)
    if ranking_norm is not None:
        return ranking_norm.rank(codes, codes.numel())[1]
    values = codes.detach().cpu().numpy()
    return np.lexsort((np.arange(len(values)), -np.abs(values))).tolist()


def reference_rows(cfg, pair, rows, context):
    """Select the reference questions separately from measured question count."""
    enabled = context is not None and context["settings"]["cohort_filter"]["enabled"]
    if not enabled and not settings(cfg)["explicit"]:
        return rows
    requested = (
        context["settings"]["cohort_filter"]["reference_sample_count"]
        if context is not None
        else cfg.task.get("sar", {}).get("cohort_filter", {}).get("reference_sample_count", 100)
    )
    if requested is None:
        return rows
    from omegaconf import OmegaConf
    from saebase.datasets.selection import select_dataset

    reference_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    reference_cfg.sample_count = requested
    reference = select_dataset(
        reference_cfg,
        model_key=pair.identity.model_key,
        model=pair.model.model,
        loaded_model_name=pair.identity.loaded_model_name,
        seed=int(cfg.seed),
    )
    observed = list(reference)
    if len(observed) != requested:
        raise ValueError(
            f"Capture feature selection requires {requested} reference questions; got {len(observed)}"
        )
    if context is not None:
        context["reference_selection"] = reference.metadata
    return observed


def fit_reference(cfg, pair, rows, context, directory):
    from tasks.capture.whitening import fit_normalization
    from tasks.capture.sar_filter import prepare_cohort_filter

    normalization, snapshots = fit_normalization(cfg, pair, rows, directory)
    if context is not None:
        prepare_cohort_filter(context, normalization, snapshots, rows)
    return normalization, {
        (row["sample_id"], row["prompt"]): snapshot for row, snapshot in zip(rows, snapshots)
    }
