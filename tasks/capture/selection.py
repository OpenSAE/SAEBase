"""Capture-owned candidate filtering and independent observation limits."""

from saebase.runtime.configuration.task_options import task_value

from tasks.capture.configuration import capture_section
from tasks.capture.sar_filter import filter_settings, cohort_filter_settings


def active_whiten(task):
    selection = capture_section(task, "selection")
    policy = selection.get("candidate_policy", "legacy")
    if policy not in {"active_whiten", "legacy"}:
        raise ValueError("task.capture.selection.candidate_policy must be active_whiten or legacy")
    return policy == "active_whiten" and type(selection.get("candidate_count")) is int


def g_s_candidate_count(cfg):
    """Task-owned Whiten screening pool; historical recipes keep their saved selection."""
    if task_value(cfg.get("task", {}), "mode") != "g_s" or active_whiten(cfg.task):
        return None
    section = capture_section(cfg.task, "g_s")
    if "candidate_count" not in section:
        return None
    value = section["candidate_count"]
    if type(value) is not int or value < 1:
        raise ValueError("task.capture.g_s.candidate_count must be a positive integer")
    return value


def settings(cfg):
    task = cfg.get("task", {})
    modern_mode = task_value(task, "mode") in {"g_s", "I"}
    selection = capture_section(task, "selection")
    explicit = "candidate_count" in selection or modern_mode
    limit = selection.get("candidate_count", 10 if modern_mode else "all")
    if limit != "all" and (type(limit) is not int or limit < 1):
        raise ValueError(
            "task.capture.selection.candidate_count must be 'all' or a positive integer"
        )
    if i_feature_selection(cfg) == "g_s":
        i_g_s_shortfall(cfg)
        if type(limit) is not int:
            raise ValueError(
                "I selection by g_s requires an integer task.capture.selection.candidate_count"
            )
    single_pool = g_s_candidate_count(cfg)
    if single_pool is not None:
        limit = single_pool
        explicit = True
    count = capture_section(task, "output").get("feature_count")
    if count is None:
        count = 10 if single_pool is not None else limit if explicit else 10
    if count != "all" and (type(count) is not int or count < 1):
        raise ValueError("task.capture.output.feature_count must be 'all' or a positive integer")
    if (
        task_value(task, "mode") not in {"sar", "I"}
        and type(limit) is int
        and type(count) is int
        and count > limit
    ):
        raise ValueError("task.capture.output.feature_count cannot exceed the candidate count")
    result = {"filter": limit, "capture_count": count, "explicit": explicit}
    if active_whiten(task):
        result["candidate_policy"] = "active_whiten"
    return result


def i_feature_selection(cfg):
    """Keep I's optional answer-based candidate ranking independent of other modes."""
    if task_value(cfg.get("task", {}), "mode") != "I" or active_whiten(cfg.task):
        return None
    value = capture_section(cfg.task, "i").get("feature_selection")
    if value not in {None, "whiten", "native_activation", "g_s"}:
        raise ValueError(
            "task.capture.i.feature_selection must be null, whiten, native_activation or g_s"
        )
    return value


def i_g_s_shortfall(cfg):
    value = capture_section(cfg.task, "i").get("g_s_shortfall", "available")
    if value not in {"available", "relax_filters"}:
        raise ValueError("task.capture.i.g_s_shortfall must be available or relax_filters")
    return value


def output_count(cfg, available):
    value = settings(cfg)["capture_count"]
    return available if value == "all" else min(value, available)


def prepare_filter(cfg):
    selection = settings(cfg)
    if selection["filter"] == "all":
        return None
    raw = capture_section(cfg.task, "selection")
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
    mode = task_value(task, "mode", "non_whiten")
    if active_whiten(task):
        return True
    default_selection = "whiten" if mode in {"g_s", "I"} else "native_activation"
    all_behavior = (
        settings(cfg)["explicit"]
        and settings(cfg)["filter"] == "all"
        and mode in {"g_s", "sar", "I"}
    )
    return (
        mode == "whiten"
        or g_s_candidate_count(cfg) is not None
        or (mode == "I" and i_feature_selection(cfg) == "whiten")
        or (
            not all_behavior
            and mode in {"g_s", "sar", "I"}
            and capture_section(task, "selection").get("feature_selection", default_selection)
            == "whiten"
        )
        or (context is not None and context["settings"]["cohort_filter"]["enabled"])
    )


def candidate_ids(cfg, pair, codes, normalization, context, options):
    import numpy as np

    value = settings(cfg)["filter"]
    mode = task_value(cfg.task, "mode", "non_whiten")
    if active_whiten(cfg.task):
        from tasks.capture.sar_filter import select_features

        if normalization is None:
            raise ValueError("Shared active Whiten selection requires reference normalization")
        return select_features(
            pair, codes, normalization, value, context, options, active_only=True
        )
    if value == "all" and mode in {"g_s", "sar", "I"}:
        return list(range(codes.numel()))
    single_pool = g_s_candidate_count(cfg)
    if single_pool is not None:
        if normalization is None:
            raise ValueError("g_s Whiten preselection requires reference normalization")
        if context is not None:
            from tasks.capture.sar_filter import select_features

            eligible = select_features(pair, codes, normalization, None, context, options)
        else:
            eligible = normalization.rank(codes, codes.numel())[1]
        return eligible[:single_pool]
    override = i_feature_selection(cfg)
    if override == "g_s":
        if type(value) is not int:
            raise ValueError(
                "I selection by g_s requires an integer task.capture.selection.candidate_count"
            )
        raw = codes.detach().cpu().numpy()
        ordered = np.lexsort((np.arange(len(raw)), -np.abs(raw))).tolist()
        active = [fid for fid in ordered if float(codes[fid]) != 0.0]
        if context is not None:
            from tasks.capture.sar_filter import select_features

            eligible = set(select_features(pair, codes, None, None, context, options))
            pool = [fid for fid in active if fid in eligible]
        else:
            pool = list(active)
        audit = {
            "fully_filtered_count": len(pool),
            "all_nonzero_count": len(active),
            "relaxed_filters": [],
        }
        if len(pool) < value and i_g_s_shortfall(cfg) == "relax_filters":
            hard = set(context.get("blocked_features", {})) if context else set()
            without_cohort = [fid for fid in active if fid not in hard]
            if len(without_cohort) > len(pool):
                pool = without_cohort
                audit["relaxed_filters"].append("cohort_frequency")
            if len(pool) < value:
                pool = list(active)
                audit["relaxed_filters"].append("semantic_and_native_density")
            if len(pool) < value:
                raise ValueError(
                    f"I needs {value} nonzero features but only {len(pool)} are active even without filters"
                )
        audit["screened_count"] = len(pool)
        if context is not None:
            context["i_g_s_pool"] = audit
        return pool
    whiten = mode == "whiten" or (
        mode in {"sar", "g_s", "I"}
        and capture_section(cfg.task, "selection").get("feature_selection", "whiten") == "whiten"
    )
    if override is not None:
        whiten = override == "whiten"
    ranking_norm = normalization if whiten else None
    if context is not None:
        from tasks.capture.sar_filter import select_features

        count = value if type(value) is int else codes.numel()
        return select_features(pair, codes, ranking_norm, count, context, options)
    if ranking_norm is not None:
        return ranking_norm.rank(codes, codes.numel())[1]
    values = codes.detach().cpu().numpy()
    return np.lexsort((np.arange(len(values)), -np.abs(values))).tolist()


def undefined_reason(result):
    """Explain an observed sample whose requested computation was undefined."""
    if result.get("candidate_shortfall"):
        return result["candidate_shortfall"]
    if result.get("sar", {}).get("status") == "undefined":
        return "SAR coalition behavior has zero variance"
    return None


def reference_rows(cfg, pair, rows, context):
    """Select the reference questions separately from measured question count."""
    enabled = context is not None and context["settings"]["cohort_filter"]["enabled"]
    if not enabled and not settings(cfg)["explicit"]:
        return rows
    requested = (
        context["settings"]["cohort_filter"]["reference_sample_count"]
        if context is not None
        else capture_section(cfg.task, "selection")
        .get("cohort_filter", {})
        .get("reference_sample_count", 100)
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


def observed_selection(selection, sample_ids):
    """Restrict observations while retaining the independently selected reference cohort."""
    if sample_ids is None:
        return selection
    if (
        isinstance(sample_ids, str)
        or not sample_ids
        or any(not isinstance(i, str) or not i for i in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("task.capture.sample_ids must be a nonempty list of unique sample IDs")
    from saebase.datasets.selection import SampleSelection

    requested = set(sample_ids)
    rows = [row for row in selection if row["sample_id"] in requested]
    missing = requested - {row["sample_id"] for row in rows}
    if missing:
        raise ValueError(
            f"Requested Capture samples absent from the selected cohort: {sorted(missing)}"
        )
    return SampleSelection(
        rows,
        {
            "source": "capture_observation_subset",
            "dataset": selection.metadata.get("dataset"),
            "selected_count": len(rows),
            "parent_selection": selection.metadata,
            "observation_sample_ids": list(sample_ids),
        },
    )
